# Data directory

**Place the datasets here**
## download corresponding datasets:
1. emotion: fer2013(20k), ferplus(20k), sfew(1k), expw(90k)
2. pose: aflw(20k)
3. gender: adience, imdb, wiki
4. age: adience(8 group), imdb
5. other attributes: celeba(40 categories)

## Folder naming and content
- Save dataset in separate folder, preserve their original directory structure
- For Adience dataset, provide 2 folder, each for images and fold.txt file
- Suggested folder structure
  ```bash
  data/
    ├── ferplus
    ├── expw
    ├── sfew
    ├── aflw
    ├── adience
    └── imdb

  ```

## Generating .csv db file in db file
Use make_db.py
usage: make_db.py [-h] --db_name {imdb,wiki,utkface,fgnet,adience} --path PATH

## search no existing file
use prep_data.py

