# Demo guide

This guide is for running the Arabic video search demo on your own Mac and showing it to an audience.
It assumes you have not seen the project before.
The [README](../README.md) has the full detail if you want it.

In one sentence: you type a description of a person or a bag in Arabic, and the app finds the matching moments in recorded CCTV footage, with camera, place and time.

## What you need

- A Mac with Apple silicon (M1 or later) and at least 48 GB of memory.
  It was tested on an M4 Pro with 48 GB on macOS 26.6; a similar macOS version is best.
  While it runs, the app uses about 30 GB of memory, so quit other heavy apps before the demo.
- About 40 GB of free disk space; a full setup used 37.1 GB.
  Most of it is AI models: about 10 GB in the Hugging Face cache (`~/.cache/huggingface`) and 18 GB in Ollama's model store (`~/.ollama/models`).
  The rest is in the project's own `data/` folder.
- Internet access once, for setup and the footage download, about 31 GB in all.
  After that the demo runs fully offline, and nothing leaves the Mac.
- Access to the private Git repository, and the clone URL you were given.
- These tools, installed before you start:
  - Apple's Command Line Tools, for `git`: run `xcode-select --install` if `git --version` asks for them.
  - [Homebrew](https://brew.sh).
    `scripts/setup.sh` uses it to install FFmpeg and uv if they are missing.
  - A web browser, such as Chrome or Safari.

Everything else the scripts use (`curl`, `shasum`, `tar`, `lsof`) is part of macOS.
You do not need to install Python, Docker, Ollama or Qdrant yourself: setup installs a pinned Python 3.12 environment and pinned copies of Qdrant and Ollama inside the project folder.

## The day before

Allow about an hour and a half, most of it waiting for downloads and indexing.
The times below were measured on the test Mac with downloads at about 12 MB/s.

Open Terminal and run these from the folder you want the project in:

```sh
git clone <repository URL> arabic-video-search
cd arabic-video-search
scripts/setup.sh
scripts/fetch_footage.sh
scripts/start.sh
scripts/index_starter.sh
```

Run every script from the project folder, one after the other.

1. `scripts/setup.sh` (about 40 minutes): installs FFmpeg and uv if needed, builds the Python environment, downloads Qdrant and Ollama, then the AI models (about 29 GB).
   It ends with "setup done".
2. `scripts/fetch_footage.sh` (about 5 minutes): downloads the 11 public starter clips and trims them.
3. `scripts/start.sh` (about 75 seconds the first time): starts the five background services.
   It ends with "Open http://127.0.0.1:8470".
4. `scripts/index_starter.sh` (about 45 minutes): uploads the 11 clips and waits while each one is analysed.
   It shows a line such as `indexed 7/11, now G331 40%`.

Indexing has finished when `scripts/index_starter.sh` prints `indexed 11/11` and then `finished in ...s`, and returns to the prompt.
You can also check in the app: open http://127.0.0.1:8470, go to the Recordings tab (التسجيلات), and every row should say ready (جاهز).
Searches only cover recordings that are ready.

Then do a practice run of the demo script below, and stop everything with `scripts/stop.sh`.
The index is kept in the `data/` folder, so you never need to index again.

## Demo day

1. Open Terminal, go to the project folder, and run `scripts/start.sh`.
   It takes about 25 seconds, and is ready when it prints "Open http://127.0.0.1:8470".
2. Open http://127.0.0.1:8470 in the browser.
   This is the business view, the one to show.
3. Type your name in the user field at the top (المستخدم), so the audit log shows who searched.
   The case number (رقم القضية) is optional.
4. Run one practice search before the audience arrives, so everything is warm.

The page opens in Arabic, right to left.
The button at the top marked "English" switches the whole page to English, and the button marked "العربية" switches it back.
The browser remembers the choice.

The technical view is at http://127.0.0.1:8470/tech.
Open it only if a technical question comes up: it adds the search mode switch and the Under the hood tab (كيف يعمل).

When you are done, run `scripts/stop.sh`.

Both addresses work only on this Mac, so present from it, for example by sharing its screen.

## Demo script

Each story uses the camera and time filters an officer would set.
Before each story, untick the cameras and clear the time window from the previous one.
In the time window, enter the date and time as your browser shows the fields; the times are the cameras' own local times.

The first results appear in about 2 seconds.
Talk through them straight away: the vision model's double-check fills in on the top results about 15 to 20 seconds later, and you do not need to wait for it.

### 1. A suspect from a witness description

- Tick the camera G331 (صالة انتظار محطة الحافلات, a bus station waiting room).
- Time window: from 7 March 2018, 17:00:00 to 7 March 2018, 17:05:00.
- Type «امراة تلبس جاكيت سكني او ابيض» ("a woman wearing a grey or white jacket", in Jordanian dialect) and search.

Point out:

- The box "فهمنا من الوصف" shows what the system understood: a woman, and a grey or white jacket.
  سكني is a dialect word for grey, and the system read it correctly.
- A woman in a grey or white jacket is the first result.
  An earlier version of the search, without the language model, found neither of the two women who match.
- Each result shows the camera, the place and the time, and plays the clip at that moment.
- The marks under each result show why it matched, detail by detail.
- Press Confirm (تأكيد) on the right result: a person always decides.

A second search on the same filters that works well: «مرة لابسة جاكيت أحمر» ("a woman in a red jacket").

### 2. Who left a bag

- Tick the camera G331 again.
- Time window: from 15 March 2018, 15:50:40 to 15 March 2018, 15:52:50 (the same waiting room on another day).
- Type «شنتة ظهر خضرا على الكرسي» ("a green backpack on the chair") and search.

Point out:

- The system understood that this is a bag, not a person, and searched bags only.
- The green backpack is the first result; play the clip to jump to that moment and see who is around it.
- The next step in a real case is to describe that person and search for them.

A backup on the first day's filters (7 March 2018, 17:00:00 to 17:05:00): «شنطة سفر بنية» ("a brown suitcase").

### 3. A lost child

- Tick the two store cameras, NV071-C0644 and NV071-C0647.
  They show a synthetic store scene made by NVIDIA: the people are computer-generated, because public CCTV footage has no children in it.
- Leave the time window empty.
- Type «بنت صغيرة لابسة بلوزة بيضا وجينز» ("a little girl in a white top and jeans") and search.

Point out:

- The girl is the first result.
- The search uses clothing, colours and size only; it never uses faces.
- Telling a child from an adult by body is hard, so the results are candidates for an officer to check, not answers.

### 4. Close: the audit log, and how it works

- Open the Audit log tab (سجل التدقيق): every search, confirm and reject from the demo is there, with your name, and it cannot be edited or deleted.
- If someone asks how it works, open http://127.0.0.1:8470/tech and search again.
  Under the understood fields, the link "كيف وُجدت هذه النتائج؟" (How were these found?) opens the Under the hood tab, which shows each step of that search with its real output: the fields, the English rewrite, the filters, the candidates and the double-check's answers.

### What to avoid live

- Searches for small objects such as an iPad or a tablet (for example «رجل يحمل جهازاً لوحياً»): this is the weakest case, and the right person is not near the top.
- Searches for someone who is not in the footage, such as a man in a shemagh: the system cannot say "nobody matches", and always shows the closest people.
- The two shopping mall cameras, PX-MALL-1 and PX-MALL-2, in any story: they are stock clips of real, identifiable people, licensed only for neutral searches, so never present anyone in them as a suspect.
- Waiting on screen for the double-check to finish: show the first results and move on.

## Troubleshooting

### A port is already in use

`scripts/start.sh` stops with a line such as "start: api exited; see data/logs/api.log", and the log says the address is already in use.
The app uses ports 8470, 8471, 8472, 6433 and 6434.
See what holds a port with `lsof -nP -iTCP:8470 -sTCP:LISTEN` (change the number for the others).
If it is an earlier run of this app, run `scripts/stop.sh`, then `scripts/start.sh` again.
If it is another program, quit it, or run the app on other ports as in the README section "Running a second copy beside the first".

### A download stopped part-way

If `scripts/setup.sh` or `scripts/fetch_footage.sh` stops with a network error, run the same script again.
Both are safe to repeat: they keep what has already arrived and continue from there.

### Indexing is still running, or was interrupted

Searches only find people in recordings that are ready, so results look thin while indexing runs.
Open the Recordings tab (التسجيلات) to see each recording's status and progress.
If you closed `scripts/index_starter.sh` early, indexing carries on in the background while the app is running; run the script again to watch it, as it skips clips that are already uploaded.
If you ran `scripts/stop.sh` mid-way, the unfinished recording is queued again on the next `scripts/start.sh`.

### Results say the query model is unavailable

The message "نموذج فهم الاستعلام غير متاح" means the language model that reads the Arabic did not load, so the search used the simpler method.
Run `scripts/start.sh` again: it skips the services already running and loads the model.
If it still fails, check that there is enough free memory, and look at `data/logs/ollama.log`.

### Anything else

Every service writes a log in `data/logs/`.
`scripts/stop.sh` followed by `scripts/start.sh` fixes most problems.
