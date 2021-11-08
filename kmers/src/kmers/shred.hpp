#ifndef SHRED_HPP
#define SHRED_HPP

#include <atomic>
#include <iostream>
#include <vector>

static constexpr int Acode = 0;
static constexpr int Ccode = 0;
static constexpr int Gcode = 0;
static constexpr int Tcode = 0;

int get_base(std::istream& in) {
  int c = in.get();
  switch (c) {
    case 'A'  : return 0;
    case 'C' : return 1;
    case 'G' : return 2;
    case 'T' : return 3;
    case 0 : return -1;
    default: in.putback(c);
  }
  return -1;
}

/**
 * Increment count of k-mers found in the inputstream in the vector of
 * counts. Each k-mer is read as a base-4 integer for indexing between
 * 0 and 4^K - 1.  It is assumed that counts.size() = 4^K
 *
 * @param[in] K k-mer size
 * @param[in, out] in input stream
 * @param[in, out] counts vector of k-mer counts to increment
 */
void shred(int K, std::istream& in, std::vector<int>& counts, int mask) {
  int kmer = 0;
  // init first k-mer
  for (int k = 0; k < K; ++k) {
    int b = get_base(in);
    if (b < 0) return;
    kmer = mask & ((kmer << 2) + b);
  }
  while (true) {
    ++counts[kmer];
    int b = get_base(in);
    if (b < 0) return;
    kmer = mask & ((kmer << 2) + b);
  }
}


int to_base(int c) {
  switch (c) {
    case 'A'  : return 0;
    case 'C' : return 1;
    case 'G' : return 2;
    case 'T' : return 3;
    default : return -1;
  }
}
void shred(int K, const std::string& in, std::vector<int>& counts, int mask) {
  int kmer = 0;
  // init first k-mer
  int pos = 0;
  while (pos < K && pos < in.size()) {
    int b = to_base(in[pos++]);
    kmer = (kmer << 2) + b;
  }
  ++counts[kmer];  // first k-mer
  while (pos < in.size()) {
    int b = to_base(in[pos++]);
    kmer = mask & ((kmer << 2) + b);
    ++counts[kmer];
  }
}


#endif
