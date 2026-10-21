"""Official DAM history: the real gridstatus parse path, validation, and source ranking.

The MIS file below has the column layout of ERCOT report NP4-190-CD (DAM Settlement
Point Prices): DeliveryDate, HourEnding "01:00".."24:00", SettlementPoint,
SettlementPointPrice, DSTFlag. It is fed through gridstatus's own parser with only the
network calls replaced, so a gridstatus parsing change would fail here.
"""
import datetime as dt
import io
import zipfile

import pandas as pd
import pytest

from src import history

gridstatus = pytest.importorskip("gridstatus")
from gridstatus.ercot import Document  # noqa: E402


def mis_zip(tmp_path, day: dt.date, hub_price=lambda he: 30 + he, extra_points=True):
    rows = ["DeliveryDate,HourEnding,SettlementPoint,SettlementPointPrice,DSTFlag"]
    for he in range(1, 25):
        for sp in ["HB_HUBAVG", "HB_BUSAVG", "HB_NORTH", "HB_HOUSTON", "HB_SOUTH", "HB_WEST", "HB_PAN"]:
            rows.append(f"{day:%m/%d/%Y},{he:02d}:00,{sp},{hub_price(he):.2f},N")
        if extra_points:
            rows.append(f"{day:%m/%d/%Y},{he:02d}:00,LZ_NORTH,{hub_price(he) + 1:.2f},N")
            rows.append(f"{day:%m/%d/%Y},{he:02d}:00,SOME_NODE_RN,{hub_price(he) + 2:.2f},N")
    p = tmp_path / f"dam_{day}.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(f"cdr.00012331.0000000000000000.{day:%Y%m%d}.133000.DAMSPNP4190.csv",
                   "\n".join(rows) + "\n")
    return p


@pytest.fixture
def fake_mis(tmp_path, monkeypatch):
    store = {}

    def _get_documents(self, report_type_id, date=None, **kw):
        out = []
        for day, path in store.items():
            pub = pd.Timestamp(day - dt.timedelta(days=1)).tz_localize("US/Central") + pd.Timedelta(hours=13, minutes=30)
            if date is not None and date != "latest" and pub.date() != date.date():
                continue
            out.append(Document(url=str(path), publish_date=pub,
                                constructed_name=f"x.{day:%Y%m%d}.csv.zip",
                                friendly_name=f"DAMSPNP4190_{day - dt.timedelta(days=1):%Y%m%d}_1330_csv",
                                friendly_name_timestamp=pub))
        if not out:
            from gridstatus.base import NoDataFoundException
            raise NoDataFoundException("none")
        return out

    monkeypatch.setattr(gridstatus.Ercot, "_get_documents", _get_documents)
    monkeypatch.setattr(gridstatus.Ercot, "_get_settlement_point_mapping",
                        lambda self, verbose=False: pd.DataFrame({"RESOURCE_NODE": ["SOME_NODE_RN"]}))

    def add(day, **kw):
        store[day] = mis_zip(tmp_path, day, **kw)
    return add


def test_gridstatus_parse_path_gives_24_hub_hours(fake_mis):
    d = dt.date(2026, 10, 21)
    fake_mis(d)
    out = history.fetch_mis_daily(d)
    hub = out[out["location"] == "HB_HUBAVG"].sort_values("he")
    assert list(hub["he"]) == list(range(1, 25))
    assert set(hub["op_date"]) == {d}
    assert hub["spp"].tolist() == [30.0 + h for h in range(1, 25)]
    assert set(out["location"]) <= set(history.HUBS)          # nodes / load zones dropped
    assert set(out["source"]) == {"ercot-mis-daily"}


def test_bad_prices_are_rejected_not_repaired(fake_mis):
    d = dt.date(2026, 10, 21)
    fake_mis(d, hub_price=lambda he: 99999.0 if he == 5 else 30.0)
    with pytest.raises(ValueError, match="outside"):
        history.fetch_mis_daily(d)


def test_refresh_stores_and_is_idempotent(fake_mis, monkeypatch):
    today = dt.date(2026, 10, 21)
    for i in range(0, 5):
        fake_mis(today - dt.timedelta(days=i))

    class FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 10, 21, 15, 0, tzinfo=tz)
    monkeypatch.setattr(history.dt, "datetime", FakeDT)
    r1 = history.refresh(days_back=4, include_tomorrow=True)
    h = history.hub_hourly()
    assert len(history.complete_days(h)) == 5
    n1 = len(history.load())
    history.refresh(days_back=4, include_tomorrow=True)
    assert len(history.load()) == n1                            # no duplicates
    assert not [f for f in r1["failed"] if "2026-10-22" in f]    # tomorrow-not-yet is normal


def test_official_source_outranks_dashboard():
    d = dt.date(2026, 10, 21)
    dash = pd.DataFrame({"op_date": [d] * 24, "he": range(1, 25), "location": "HB_HUBAVG",
                         "spp": 10.0, "source": "ercot-dashboard"})
    mis = dash.assign(spp=11.0, source="ercot-mis-daily")
    history.upsert(mis)
    history.upsert(dash)                                        # later, but lower authority
    h = history.hub_hourly()
    assert (h["spp"] == 11.0).all()


def test_validation_drops_partial_days():
    d = dt.date(2026, 10, 21)
    full = pd.DataFrame({"op_date": [d] * 24, "he": range(1, 25), "location": "HB_HUBAVG",
                         "spp": 30.0, "source": "x"})
    part = pd.DataFrame({"op_date": [d + dt.timedelta(days=1)] * 10, "he": range(1, 11),
                         "location": "HB_HUBAVG", "spp": 30.0, "source": "x"})
    out = history.validate(pd.concat([full, part]), "t")
    assert set(out["op_date"]) == {d}
