- **Layer streaming on Windows now checks system commit headroom before it page-locks the disk tier's host staging (#PRNUM5).**
  Page-locked memory is charged against the commit limit (RAM + pagefile), which the free-RAM
  pre-flight does not show, so a 32 GB laptop with RAM to spare could be stopped mid-run (a 123B
  run was, twice, with pinned staging; pageable staging passed). With `training.stream_pin` unset
  and less than 8 GiB of commit left after the staging, Soup now stages in pageable memory and
  says so in the panel; `stream_pin: true` is refused before any weight is staged when the
  staging cannot fit the commit limit, naming the numbers and the fix (`stream_pin: false`,
  close programs, a larger pagefile). `stream_pin: false` and every other OS are unchanged, and
  a failed read of the counters skips the check.
