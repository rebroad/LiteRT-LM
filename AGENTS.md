# LiteRT-LM repo notes

## Repo boundary
- This repo owns the runtime, engine, session, and model execution behavior.
- Do not keep the Android Gallery HTTP server implementation here.
- The Gallery app owns `OpenAiHttpServer.kt` and the phone server UI.

## Gallery dependency
- Gallery consumes LiteRT-LM as a built artifact.
- If a runtime fix is needed in Gallery, change LiteRT-LM here, rebuild the artifact, then reinstall Gallery against that artifact.
- If Gallery appears to ignore a runtime change, assume it is still using an older published artifact until proven otherwise.

## Practical rule
- Multi-session / engine behavior belongs here.
- HTTP server presentation and app-side request wiring belong in Gallery.
- Avoid duplicate HTTP server code across the repos unless you are temporarily cherry-picking a porting commit, and delete the duplicate afterward.
