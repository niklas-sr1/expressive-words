#!/usr/bin/env python3

import random
import sys

CHOICES = 5

with open("expressive_words.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f.readlines()]

if len(sys.argv) > 1:
    arg_version = sys.argv[1].lstrip("v")
else:
    arg_version = "0.x.x"

for i in range(CHOICES):
    random_word = random.choice(words)
    debug_version = f"v{arg_version}-debug.id-{random_word}"

    print(debug_version)
