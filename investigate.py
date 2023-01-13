import json

if __name__ == "__main__":
    with open("data/scene_datasets/replica/room_0/habitat/info_semantic.json") as f:
        obj = json.load(f)

    print(obj.keys())
    print(obj['classes'])
    print(len(obj['classes']))
