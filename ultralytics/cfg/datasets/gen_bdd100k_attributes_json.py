"""
BDD100K JSON -> attributes.json generation script

output:
{
    "b1c66a42-6f7d68ca": {"weather": 0, "timeofday": 0, "scene": 0},
    ...
}
- undefined -> -1 (ignore_index for CrossEntropyLoss)

Usage:
  python ./ultralytics/cfg/datasets/gen_bdd100k_attributes_json.py \
    --input  /media/data/bdd100k_yolo/bdd100k_labels_images_train.json \
    --output /media/data/bdd100k_yolo/train/attributes.json \
    2>&1 | tee ./ultralytics/cfg/datasets/gen_bdd100k_train_attributes.log
  
  python ./ultralytics/cfg/datasets/gen_bdd100k_attributes_json.py \
    --input  /media/data/bdd100k_yolo/bdd100k_labels_images_val.json \
    --output /media/data/bdd100k_yolo/val/attributes.json \
    2>&1 | tee ./ultralytics/cfg/datasets/gen_bdd100k_val_attributes.log
"""

import argparse
import json
from pathlib import Path

# class mapping
WEATHER_MAP = {
    "clear":         0,
    "rainy":         1,
    "snowy":         2,
    "overcast":      3,
    "foggy":         4,
    "partly cloudy": 5,
    "undefined":    -1,  # ignore_index
}

TIMEOFDAY_MAP = {
    "daytime":   0,
    "night":     1,
    "dawn/dusk": 2,
    "undefined": -1,
}

SCENE_MAP = {
    "city street": 0,
    "highway":     1,
    "residential": 2,
    "parking lot": 3,
    "tunnel":      4,
    "gas stations":5,
    "undefined":  -1,
}


def main():
    parser = argparse.ArgumentParser(description="BDD100K -> attributes.json generation")
    parser.add_argument("--input",  type=str, required=True, help="BDD100K JSON file path")
    parser.add_argument("--output", type=str, required=True, help="attributes.json output path")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    attributes = {}
    unknown = {"weather": set(), "timeofday": set(), "scene": set()}

    for frame in data:
        stem = Path(frame["name"]).stem  # "b1c66a42-6f7d68ca"
        attr = frame.get("attributes", {})

        weather   = attr.get("weather", "undefined")
        timeofday = attr.get("timeofday", "undefined")
        scene     = attr.get("scene", "undefined")

        if weather not in WEATHER_MAP:
            unknown["weather"].add(weather)
        if timeofday not in TIMEOFDAY_MAP:
            unknown["timeofday"].add(timeofday)
        if scene not in SCENE_MAP:
            unknown["scene"].add(scene)

        # unknown values -> -1 (ignore_index)
        attributes[stem] = {
            "weather":   WEATHER_MAP.get(weather, -1),
            "timeofday": TIMEOFDAY_MAP.get(timeofday, -1),
            "scene":     SCENE_MAP.get(scene, -1),
        }
        
        
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(attributes, f)
    
    undefined_count = sum(
        1 for v in attributes.values()
        if -1 in v.values()
    )
 
    print(f"Done!")
    print(f"  total     : {len(data)}")
    print(f"  saved     : {len(attributes)}")
    print(f"  undefined : {undefined_count}")
    if any(unknown.values()):
        print(f"  unknown values (mapped to -1):")
        for key, vals in unknown.items():
            if vals:
                print(f"    {key}: {vals}")
    print(f"  saved to  : {args.output}")
 
 
if __name__ == "__main__":
    main()