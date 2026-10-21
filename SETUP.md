# Setup — browser only, about 10 minutes

No Terminal, no GitHub Desktop, no Finder. Everything happens on github.com.

## 1. Create the repository
1. Go to **github.com/new**
2. Name: `ercot-forecast` · **Public** · do **not** tick "Add a README"
3. **Create repository**

## 2. Upload the files
1. On the new page click **uploading an existing file**.
2. Unzip `ercot-forecast.zip`. Open the folder and drag **everything inside it** onto the browser window: `src`, `tests`, `docs`, `data`, `forecasts`, `outcomes`, `notes`, `backtest`, `README.md`, `SETUP.md`, `requirements.txt`, `.gitignore`.
3. Click **Commit changes**.

## 3. Create the schedule file (the one macOS hides)
1. **Add file → Create new file**
2. Filename, typed exactly, slashes included: `.github/workflows/daily.yml`
3. Open **`daily.yml.txt`** (it's next to the zip), select all, copy, and paste it into the editor.
4. **Commit changes** → **Commit changes**
5. Do the same again for `.github/workflows/test.yml`, pasting from **`test.yml.txt`**.

## 4. Add your EIA key
**Settings → Secrets and variables → Actions → New repository secret.** Name: `EIA_API_KEY`. Value: your key ([free here](https://www.eia.gov/opendata/register.php)).

## 5. Let it save its own work ⚠️
**Settings → Actions → General → Workflow permissions → Read and write permissions → Save.**
If you skip this, the job runs and then fails with a bare `403` when it tries to save.

## 6. Turn on the website
**Settings → Pages → Deploy from a branch → `main` / `/docs` → Save.**
The site appears at `https://AAaali2292.github.io/ercot-forecast/` within a few minutes.

## 7. Run it, in this order
**Actions** tab → enable workflows if asked → **ERCOT daily forecast** → **Run workflow** ▾ → pick from the list:

1. `status`: every line should say **OK**. If ERCOT shows FAIL with 403, just run it again (see below).
2. `backfill`: pulls about 200 days of official day-ahead prices (about 3 minutes).
3. `daily`: makes the first forecast. **Run this before 10:00 CT (20:00 Pakistan time)**, or it won't count as a live call.
4. `backtest`: fills the simulated backtest section (about 5 minutes).

After that, everything runs by itself.

## What happens each day
- **~06:07 CT (16:07 PKT):** tomorrow's forecast is committed.
- **~14:00 CT (00:00 PKT):** ERCOT publishes the cleared prices, and the call is scored.
- The site and the README table update themselves.

## If something goes wrong
| Symptom | Fix |
|---|---|
| Red X, log mentions `403` on push | Step 5 |
| `EIA-930 ... WARN no EIA_API_KEY` | Step 4: the name must be exactly `EIA_API_KEY` |
| `ERCOT ... 403` | ERCOT blocked that runner's address. Run again; GitHub runners are US-based and normally pass. |
| `Open-Meteo previous-day forecasts FAIL` | The model falls back and says so in the forecast file. The backtest refuses to run without it, by design. |
| Pages shows 404 | Step 6 (`main` + `/docs`); allow 5 minutes |

You never need to touch the data files. To change an assumption, edit `src/config.py` on github.com (pencil icon → Commit).
