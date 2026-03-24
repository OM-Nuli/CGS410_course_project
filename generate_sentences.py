import os
import torch
import csv
import random
from transformers import pipeline, AutoTokenizer

def main():
    OUTPUT_DIR = "generated_sentences"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading model {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    generator = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    
    prompts = {
        "english": [
            "Write a simple sentence about daily life.",
            "Write a funny sentence about daily life.",
            "Write a sad sentence about daily life.",
            "Describe a daily activity in one short sentence.",
            "Write a sentence about morning routine.",
            "Write a random everyday observation.",
            "Write a sentence using 'I' about daily life.",
            "Write a sentence using 'They' about daily life.",
            "Write a question about daily life.",
            "Write a surprising sentence about daily life.",
            "Write a sentence starting with 'Although'.",
            "Write a sentence without using 'I'.",
            "Write a sentence about an object, not a person.",
            "Write a metaphor about daily life.",
            "Write a sentence in passive voice."
        ],

        "chinese": [
            "写一个关于日常生活的简单句子。",
            "写一个有趣的日常句子。",
            "写一个悲伤的日常句子。",
            "用一句话描述一个日常活动。",
            "写一个关于早晨的句子。",
            "写一个日常观察。",
            "用“我”写一句话。",
            "写一个问题句。",
            "写一个令人惊讶的句子。",
            "写一个隐喻句子。"
        ],

        "vietnamese": [
            "Viết một câu đơn giản về cuộc sống hàng ngày.",
            "Viết một câu hài hước về cuộc sống.",
            "Viết một câu buồn.",
            "Mô tả một hoạt động hằng ngày.",
            "Viết một câu hỏi về cuộc sống.",
            "Viết một câu bất ngờ."
        ],

        "thai": [
            "เขียนประโยคง่ายๆ เกี่ยวกับชีวิตประจำวัน",
            "เขียนประโยคตลก",
            "เขียนประโยคเศร้า",
            "อธิบายกิจกรรมประจำวัน",
            "เขียนคำถามเกี่ยวกับชีวิต"
        ],

        "indonesian": [
            "Tulis kalimat sederhana tentang kehidupan sehari-hari.",
            "Tulis kalimat lucu.",
            "Tulis kalimat sedih.",
            "Deskripsikan aktivitas harian.",
            "Tulis kalimat pertanyaan."
        ],

        "wolof": [
            "Bind ab jëf bu nees ci mbir yi.",
            "Bind ab jëf bu neex.",
            "Bind ab jëf bu metti.",
            "Bind ab laaj ci mbir yi."
        ]
    }

    target_count = 1000

    for lang, prompt_list in prompts.items():
        print(f"\n--- Generating {lang.upper()} ---")
        sentences = set()

        while len(sentences) < target_count:

            base_instruction = random.choice(prompt_list)

            # 🔥 Ultra-advanced diversity prompt
            prompt = f"""
Generate ONE unique sentence in {lang}.

Requirements:
- Use a different structure than usual
- Vary subject (person, object, place, idea)
- Vary tone (happy, sad, funny, neutral)
- Avoid repeating common phrases
- Keep it natural and human-like
- Do NOT repeat previous patterns

Instruction:
{base_instruction}

Answer:
"""

            outputs = generator(
                prompt,
                max_new_tokens=40,
                num_return_sequences=10,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

            for output in outputs:
                raw_text = output['generated_text']

                # Remove prompt
                generated_sentence = raw_text.replace(prompt, "").strip()

                # Take first line only
                clean_sentence = generated_sentence.split('\n')[0].strip()

                if len(clean_sentence) > 5:
                    sentences.add(clean_sentence)

            print(f"{lang}: {len(sentences)}/{target_count}", end="\r")

        final_sentences = list(sentences)[:target_count]

        output_file = f"{OUTPUT_DIR}/{lang}_generated.csv"
        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["language", "sentence"])

            for sentence in final_sentences:
                writer.writerow([lang, sentence])

        print(f"\n✅ Saved → {output_file}")

if __name__ == "__main__":
    main()