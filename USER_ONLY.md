**“Truncate” means creating a shorter copy of each sequence. Your original files stay unchanged.**

Suppose one sequence contains 50 frames:

```
Original sequence:  frames 1–50
Shortened copy:     frames 1–20
```

The script sorts image filenames, copies the first 20 images, and copies their matching label files. That is why names like `frame_000001.jpg` are useful: filename order matches chronological order.

**It takes the first 20 frames automatically—it does not search for smoke.** If smoke only appears at frame 30, the shortened copy will contain no smoke. Prepare your positive clips so the retained portion includes actual smoke.

The other detail concerns rerunning the script:

1. You run it once, creating a shortened sequence.
2. You later correct a box annotation in the original sequence.
3. You run it again with the same output directory.
4. Because that shortened sequence’s folder already exists, the script **skips the entire sequence**. The copied annotation remains outdated.

To regenerate it, use a new output directory, for example:

```
First run:    data/01_raw/my_sequences_truncated/
Updated run:  data/01_raw/my_sequences_truncated_v2/
```

Then point the tube-building command at the new directory. **A new directory forces fresh copies of your corrected images and labels.**
