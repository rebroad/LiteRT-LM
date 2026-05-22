# LiteRT-LM repo notes

## Repo split
- LiteRT-LM owns runtime, engine, session, model exec.
- Gallery owns `OpenAiHttpServer.kt` and server UI.
- Do not keep Android HTTP server code here.

## Gallery on phone
- Gallery uses built LiteRT-LM artifact, not source tree.
- On Android, use Android AAR / Android arm64 output, not host JVM jar.
- Local phone build needs both:
  - `litertlm-jvm.jar`
  - `liblitertlm_jni.so`
- If Gallery ignores a runtime change, assume old artifact until proven otherwise.
- For phone rebuild/install, the Gallery script is the reminder.

## Rules
- Runtime fix -> LiteRT-LM.
- App HTTP wiring -> Gallery.
- No duplicate HTTP server copies.
- Do not assume user wants less feature just because it is annoying.
- If you find bad assumption or repo-boundary mistake, update AGENTS.md right away.
- Write notes short. Caveman short.
