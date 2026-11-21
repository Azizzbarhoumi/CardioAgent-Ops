# Demo samples (git-ignored WAV files)

The API's `GET /samples` and `GET /samples/{id}` endpoints serve the WAV files listed in `samples.json`.
The WAV files are NOT committed. To recreate them from the local dataset:

```bash
cp Data/dataset/physionet/training-a/a0238.wav samples/physionet_a0238_normal.wav
cp Data/dataset/physionet/training-a/a0296.wav samples/physionet_a0296_abnormal.wav
```

Labels follow PhysioNet's official `REFERENCE.csv` (-1 = normal, 1 = abnormal).
Source: PhysioNet/CinC Challenge 2016 (Clifford et al.). Check the dataset licence before redistributing.
Set `SAMPLES_DIR` to use another folder (default: `samples`).
