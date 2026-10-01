- Terminal output strips the C1 control range (U+0080-U+009F) as well as C0
  and DEL, and text read from files is printed through the shared
  `for_terminal` helper instead of raw, through `escape()` alone, or as
  `{escape(x)!r}`: adapter-config base model names in `soup chat`, `soup
  merge`, `soup serve`, `soup export` and the `--trust-remote-code` warning;
  `soup adapters list` / `info` / `compare` / `diff`; benchmark names in the
  `soup ship` footer; `soup edit diff`; the SFT "cannot open image/audio"
  warnings; the config loader's validation lines; `soup probe interference`
  loss keys; and the Web UI's log line for a refused config. A signature
  sidecar whose `backend` name contains markup now gets `soup attest
  verify`'s documented exit code 3 instead of 1, a losses key containing
  markup gets `soup probe interference`'s exit code 2, and a `.can` manifest
  `author` refuses control characters (#PR).
