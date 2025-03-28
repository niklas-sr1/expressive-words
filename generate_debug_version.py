#!/usr/bin/env python3

import random

CHOICES = 5

with open("expressive_words.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f.readlines()]

for i in range(CHOICES):
    random_word = random.choice(words)
    debug_version = f"v0.x.x-debug.id-{random_word}"

    print(debug_version)
