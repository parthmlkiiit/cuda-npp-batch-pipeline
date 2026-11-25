# data

Images are not committed. Get them with

```bash
./scripts/download_data.sh      # or: make data
```

This downloads the USC-SIPI `misc`, `textures`, `aerials` and `sequences`
volumes into `data/raw/` and converts them to PNG in `data/sipi/`
(210 images, 256x256 up to 2250x2250).

Any other folder of png/jpg/bmp/pgm images works too:
`./bin/npp_batch --input_dir=path/to/images`.
