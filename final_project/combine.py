import glob
import os


def combine_and_truncate_per_file(input_folder, output_filename, limit):
    filenames = glob.glob(os.path.join(input_folder, "*.txt"))

    if not filenames:
        print(f"No files found in {input_folder}")
        return

    # Open the single output file for the whole corpus
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for fname in filenames:
            file_word_count = 0

            with open(fname, 'r', encoding='utf-8') as infile:
                for line in infile:
                    words = line.split()
                    line_len = len(words)

                    if file_word_count + line_len <= limit:
                        outfile.write(line)
                        file_word_count += line_len
                    else:
                        # Grab the exact remaining words to hit the limit
                        words_needed = limit - file_word_count
                        if words_needed > 0:
                            outfile.write(" ".join(words[:words_needed]) + "\n")
                        break  # Stop this file, move to the next one

            print(f"Added {file_word_count} words from {os.path.basename(fname)}")

    print(f"--- Finished! Created {output_filename} ---")


# Run it
word_limit = 727500 # per input file
combine_and_truncate_per_file("Corpus_A_clean", "corpus_A.txt", word_limit)
combine_and_truncate_per_file("Corpus_B_clean", "corpus_B.txt", word_limit)
