# Fixed-native-K timing rule (publication transcription)

This text transcribes the original freeze.json; this Markdown file was prepared
for publication after the experiment. The freeze is the pre-run record.

Primary sequence 2048; secondary 4096. Keep native K unchanged in both arms.
Five fresh-process AB/BA pairs per sequence, in the exact roster order recorded in
freeze.json. Ten warmups (the first two replay the native reference exactly),
then 50 measured updates per process. The shared allocator and native override
are 8,589,934,592 bytes. Exactness, memory and resource gates must all pass.

Median of five paired percentage time reductions must be >=2%, with ordinary
95% and family-adjusted 97.5% bootstrap lower bounds >0. The two-cell family
retains both sequences, with no replacement of a failed primary by its secondary.
10,000 whole-pair resamples, NumPy default_rng(42), linear quantiles.

Frozen decision: median >= 2 percent and both lower bounds > 0, all exactness/memory/resource gates; secondary never replaces primary
