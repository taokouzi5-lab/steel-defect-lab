# Sample images

12 defect images (2 per class, 6 classes) copied from the **NEU-DET public
dataset test split** (via `training/manifest.json`, entries with
`"split": "test"`), served by `GET /api/v1/samples` and
`POST /api/v1/predict/sample/{sample_id}` for demos and smoke tests.

- Source: K. Song, Y. Yan, "A noise robust method based on completed local
  binary patterns for hot-rolled steel strip surface defects recognition",
  Applied Surface Science 285 (2013), pp. 858-864. Public research dataset.
- `manifest.json` is the server-side whitelist for sample ids. The API never
  joins user input into filesystem paths: a request can only reference an id
  present in this manifest.
