import os
import torch
import csv
from transformers import pipeline

def main():
    OUTPUT_DIR = "generated_sentences"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading model {model_id}...")

    generator = pipeline(
        "text-generation",
        model=model_id,
        device=0 if torch.cuda.is_available() else -1,
        pad_token_id=50256
    )

    prompts = {
        "english": "Write a simple sentence about daily life.",
        "chinese": "写一个关于日常生活的简单句子。",
        "vietnamese": "Viết một câu đơn giản về cuộc sống hàng ngày.",
        "thai": "เขียนประโยคง่ายๆ เกี่ยวกับชีวิตประจำวัน",
        "indonesian": "Tulis kalimat sederhana tentang kehidupan sehari-hari.",
        "wolof": "Bind ab jëf bu nees ci mbir yi."
    }

    target_count = 1000

    for lang, prompt in prompts.items():
        print(f"\n--- Generating {lang.upper()} ---")
        sentences = set()

        while len(sentences) < target_count:
            outputs = generator(
                prompt,
                max_new_tokens=30,
                num_return_sequences=10,
                do_sample=True,
                temperature=0.7,   # 🔥 lower = cleaner output
                top_p=0.9
            )

            for output in outputs:
                raw_text = output['generated_text']

                # Remove prompt
                generated_sentence = raw_text.replace(prompt, "").strip()

                # Take only first sentence/line
                clean_sentence = generated_sentence.split('\n')[0].strip()

                # Basic filter (DO NOT over-clean)
                if len(clean_sentence) > 5:
                    sentences.add(clean_sentence)

            print(f"{lang}: {len(sentences)}/{target_count}", end="\r")

        final_sentences = list(sentences)[:target_count]

        # ✅ Save as CSV (FIXED ENCODING)
        output_file = f"{OUTPUT_DIR}/{lang}_generated.csv"
        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            
            writer.writerow(["language", "sentence"])
            
            for sentence in final_sentences:
                writer.writerow([lang, sentence])

        print(f"\n✅ Saved → {output_file}")

if __name__ == "__main__":
    main()