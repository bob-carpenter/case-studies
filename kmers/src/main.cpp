#include "kmers/fasta-parser.hpp"
#include <iostream>
#include <vector>

struct shredder {
  const int64_t K_;
  std::vector<int32_t> count_;
  shredder(int64_t K) : K_(K), count_(fasta::kmers(K), 0) { }
  void increment(const std::string& kmer) {
    int id = fasta::kmer_id(kmer);
    ++count_[id];
  }
  void operator()(const std::string& id, const std::string& seq) {
    for (int start = 0; start < seq.size() - K_ + 1; ++start) {
      std::string kmer = seq.substr(start, K_);
      increment(kmer);
    }
  }
};

int main() {
  // std::string file = "../data/unpacked/refseq-select-2020-10-22.fasta";
  std::string file = "../data/unpacked/GRCh38_latest_rna.fna";

  std::cout << std::endl;
  std::cout << "parsing fasta file = " << file
            << std::endl;

  size_t num_targets = 0;
  size_t num_bases = 0;
  auto callback_handler
      = [&](const std::string& id, const std::string& seq) {
    ++num_targets;
    num_bases += seq.size();
    if ((num_targets % 50000) == 0)
      std::cout << "     # PARSER: "
                << (num_targets / 1000) << "K targets processed"
                << std::endl;
  };
  std::size_t K = 10;
  shredder f(K);
  fasta::parse_file(file, callback_handler);

  std::cout << (num_targets / 1000) << "K targets"
            << std::endl
            << (num_bases / 1000000) << "M bases"
            << std::endl
            << std::endl;

  return 0;
}
