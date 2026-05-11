import json

def convert_scicite(input_path, output_path):
    converted = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            new_item = {
                "text": data["string"],
                "label": data["label"],
                "section": data["sectionName"],
                "worthiness": data["label_confidence"]
            }

            converted.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"转换完成: {len(converted)} 条数据")


if __name__ == "__main__":
    convert_scicite(
        r"D:\reproduce\scicite\train.jsonl",
        r"D:\reproduce\scicite\train_converted.json"
    )

    convert_scicite(
        r"D:\reproduce\scicite\dev.jsonl",
        r"D:\reproduce\scicite\dev_converted.json"
    )

    convert_scicite(
        r"D:\reproduce\scicite\test.jsonl",
        r"D:\reproduce\scicite\test_converted.json"
    )