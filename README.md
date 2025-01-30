# Poly2Vec_GeoAI

### Installation
1. We recommend using a conda environment with Python >= 3.9:
```
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
### Preprocessed Data
1. We use two OSM datasets Singapore and NewYork. The original datasets can be found in []. 
2. After downloading the two datasets into "./data/", run ```python utils/data_preprocessing``` which will give you "poi_normalized.pkl", "roads_normalized.pkl" and "buildings_normalized.pkl" in each dataset's folder.
3. Then run ```python utils/data_preprocessing``` to generate the datasets for each tasks. You'll find files such as "polygon_polygon_topological_relationship_data.pt" in each dataset's folder.

### Train Poly2Vec
1. All the training hyperparameters can be set in config.json.
2. To train Poly2Vec for a specific task, e.g., the polygon-polygon topological classification of NewYork dataset, run
```
python run.py -dataset_name "NewYork" -dataset_type1 "polygons" -dataset_type2 "polygons" -task "multi-relation" -data_file "./data/NewYork/polygon_polygon_intersect_data.pt" -encoder_type "poly2vec" -data_path "./data/NewYork" -num_classes 6 
```
