#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "MersenneTwister.h"

int main(int argc, char ** argv)
{
  FILE * fp;
  char * wordList;
  int * wordLocs, * wordLens;
  int wordListLen, numWords;

  if (argc != 5)
  {
    printf("Usage: %s output_file file_size_in_mb avg_words_per_line random_seed.\n", *argv);
    fflush(stdout);
    return 1;
  }
  // off_t len;
  const char * const outFile  = argv[1];
  const int outputBytes       = atoi(argv[2]) * 1048576;
  const int meanWordsPerLine  = atoi(argv[3]);
  const int randomSeed        = atoi(argv[4]);
  MTRand mtrand(randomSeed);
  printf("Writing file '%s' of size %d MB with %d words per line (on average) and a random seed of %d.\n",
         outFile, outputBytes / 1048576, meanWordsPerLine, randomSeed);
  fflush(stdout);

  fp = fopen("data/wordcount/wordlist", "rb");
  fseek(fp, 0, SEEK_END);
  wordListLen = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  wordList = new char[wordListLen + 1];
  if (fread(wordList, wordListLen, 1, fp) != 1)
  {
    printf("Error reading.\n");
  }
  fclose(fp);
  wordList[wordListLen] = '\0';

  int currentWord;
  numWords = 0;
  for (int i = 0; i < wordListLen; ++i) if (wordList[i] == '\n') ++numWords;
  wordLocs = new int[numWords];
  wordLens = new int[numWords];
  currentWord = 0;
  wordLocs[currentWord++] = 0;
  for (int i = 1; i < wordListLen; ++i) if (wordList[i - 1] == '\n') wordLocs[currentWord++] = i;
  for (int i = 0; i < numWords;    ++i) wordLens[i] = wordLocs[i + 1] - wordLocs[i] - 1;
  wordLens[numWords - 1] = wordListLen - wordLocs[numWords - 1];

  /*
    file format is:
    <number of words = 4 bytes>
    <number of lines = 4 bytes>
    <empty space = 120 bytes>
    foreach line
      <offset of line = 4 bytes>
    foreach line
      <length of line = 4 bytes>
    <each line = variable number of bytes>
  */

  //  a few optimizations we're making
  //    no line is longer than 1024 bytes.
  //    every line (which includes its LF character) is padded to the nearest four bytes.
  //    we generate 32 lines at a time
  //      if not all lines would fit into the file, we don't put any of them in the file

  const int THREADS_PER_WARP = 32;
  int wordCount = 0;
  int lineCount = 0;
  int numWordsForLines[THREADS_PER_WARP];
  std::vector<int> wordsForLines[THREADS_PER_WARP];
  std::vector<int> lineOffsets, lineLengths;
  int fileSize = 128; // initial header size
  char * output = new char[outputBytes], * ptr;
  ptr = output;
  while (fileSize < outputBytes)
  {
    int chunkSize = sizeof(int) * 2 * 32;
    // int newLineOffsets[32], newLineLengths[32];
    for (int i = 0; i < THREADS_PER_WARP; ++i)
    {
      numWordsForLines[i] = mtrand.randInt(meanWordsPerLine * 2 - 1) + 1;
      wordsForLines[i].resize(numWordsForLines[i]);
      for (int j = 0; j < numWordsForLines[i]; ++j)
      {
        wordsForLines[i][j] = mtrand.randInt(numWords - 1);
        chunkSize += wordLens[wordsForLines[i][j]] + 1;
      }
      if (chunkSize % 4 != 0) chunkSize += 4 - chunkSize % 4;
    }
    if (fileSize + chunkSize > outputBytes)
    {
      // printf("Adding %d bytes of fluff.\n", outputBytes - fileSize);
      ptr += outputBytes - fileSize;
      fileSize = outputBytes;
      break;
    }
    for (int i = 0; i < THREADS_PER_WARP; ++i)
    {
      lineOffsets.push_back(ptr - output);
      int countForThisLine = 0;
      for (int j = 0; j < numWordsForLines[i]; ++j)
      {
        memcpy(ptr, wordList + wordLocs[wordsForLines[i][j]], wordLens[wordsForLines[i][j]]);
        ptr += wordLens[wordsForLines[i][j]];
        *(ptr++) = ' ';
        countForThisLine += wordLens[wordsForLines[i][j]] + 1;
      }
      while (countForThisLine % 4 != 0)
      {
        *(ptr++) = ' ';
        ++countForThisLine;
      }
      *(ptr - 1) = '\n';
      lineLengths.push_back(countForThisLine);
      wordCount += numWordsForLines[i];
    }
    lineCount += THREADS_PER_WARP;
    fileSize += chunkSize;
  }
  int headerLength = 128 + lineOffsets.size() * sizeof(int) * 2;
  for (unsigned int i = 0; i < lineOffsets.size(); ++i) lineOffsets[i] += headerLength;
  fp = fopen(outFile, "wb");
  char blank[120] = { 0 };
  int totalSize = 0;
  // printf("writing %d words, %d lines, and %d bytes of character output.\n", wordCount, lineCount, ptr - output);
  if (fwrite(&wordCount,      sizeof(wordCount),        1, fp) != 1) printf("Error writing at line %d.\n", __LINE__); totalSize += sizeof(wordCount);
  if (fwrite(&lineCount,      sizeof(lineCount),        1, fp) != 1) printf("Error writing at line %d.\n", __LINE__); totalSize += sizeof(lineCount);
  if (fwrite(blank,           sizeof(blank),            1, fp) != 1) printf("Error writing at line %d.\n", __LINE__); totalSize += sizeof(blank);
  if (fwrite(&lineOffsets[0], sizeof(int) * lineCount,  1, fp) != 1) printf("Error writing at line %d.\n", __LINE__); totalSize += sizeof(int) * lineCount;
  if (fwrite(&lineLengths[0], sizeof(int) * lineCount,  1, fp) != 1) printf("Error writing at line %d.\n", __LINE__); totalSize += sizeof(int) * lineCount;
  if (fwrite(output, ptr - output, 1, fp) != 1) printf("Error writing at line %d.\n", __LINE__);                      totalSize += ptr - output;
  fflush(stdout);
  fclose(fp);
  printf("wrote %d words, %d lines, and %d bytes of character output, for a total of %.3f MB.\n", 
         wordCount, 
         lineCount, 
         static_cast<int>(ptr - output), 
         static_cast<double>(totalSize) / 1048576.0);

  return 0;
}
