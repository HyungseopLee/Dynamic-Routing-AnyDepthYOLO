"""
BDD100K JSON -> attributes.json generation script

output:
{
    "b1c66a42-6f7d68ca": {"weather": 0, "timeofday": 0, "scene": 0},
    ...
}


Usage:
  python ./ultralytics/cfg/datasets/gen_bdd100k_attributes_json.py \
    --input  /media/data/bdd100k_yolo/bdd100k_labels_images_train.json \
    --output /media/data/bdd100k_yolo/train/attributes.json \
    2>&1 | tee ./ultralytics/cfg/datasets/gen_bdd100k_train_attributes.log
  
  python ./ultralytics/cfg/datasets/gen_bdd100k_attributes_json.py \
    --input  /media/data/bdd100k_yolo/bdd100k_labels_images_val.json \
    --output /media/data/bdd100k_yolo/train/attributes.json \
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
}

TIMEOFDAY_MAP = {
    "daytime":   0,
    "night":     1,
    "dawn/dusk": 2,
}

SCENE_MAP = {
    "city street": 0,
    "highway":     1,
    "residential": 2,
    "parking lot": 3,
    "tunnel":      4,
    "gas stations":5,
}


def main():
    parser = argparse.ArgumentParser(description="BDD100K → attributes.json 생성")
    parser.add_argument("--input",  type=str, required=True, help="BDD100K JSON 파일 경로")
    parser.add_argument("--output", type=str, required=True, help="attributes.json 저장 경로")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    attributes = {}
    skipped = 0
    unknown = {"weather": set(), "timeofday": set(), "scene": set()}

    for frame in data:
        stem = Path(frame["name"]).stem  # "b1c66a42-6f7d68ca"
        attr = frame.get("attributes", {})

        weather   = attr.get("weather", "")
        timeofday = attr.get("timeofday", "")
        scene     = attr.get("scene", "")

        if weather not in WEATHER_MAP:
            unknown["weather"].add(weather)
        if timeofday not in TIMEOFDAY_MAP:
            unknown["timeofday"].add(timeofday)
        if scene not in SCENE_MAP:
            unknown["scene"].add(scene)

        if weather not in WEATHER_MAP or timeofday not in TIMEOFDAY_MAP or scene not in SCENE_MAP:
            skipped += 1
            continue

        attributes[stem] = {
            "weather":   WEATHER_MAP[weather],
            "timeofday": TIMEOFDAY_MAP[timeofday],
            "scene":     SCENE_MAP[scene],
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(attributes, f)

    print(f"Done!")
    print(f"  total   : {len(data)}")
    print(f"  saved   : {len(attributes)}")
    print(f"  skipped : {skipped}")
    if any(unknown.values()):
        print(f"  unknown values:")
        for key, vals in unknown.items():
            if vals:
                print(f"    {key}: {vals}")
    print(f"  saved to: {args.output}")


if __name__ == "__main__":
    main()