#include "kmers/fasta-parser.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <locale>
#include <vector>

struct number_format : public std::numpunct<char> {
  char do_thousands_sep() const { return ','; }
  std::string do_grouping() const { return "\03"; }
};

bool valid_base(char c) {
    switch (c) {
    case 'A':
    case 'C':
    case 'G':
    case 'T': return true;
    default: return false;
    }
}

bool valid_bases(const std::string& seq) {
  return std::all_of(seq.begin(), seq.end(), valid_base);
  //   for (char c : seq)
  // if (!valid_base(c))
  // return false;
  // return true;
}

template <typename F1, typename F2>
struct coupler {
  F1& f1_;
  F2& f2_;
  coupler(F1& f1, F2& f2) : f1_(f1), f2_(f2) { }
  void operator()(const std::string& id, const std::string& seq) {
    f1_(id, seq);
    f2_(id, seq);
  }
  void report() {
    f1_.report();
    f2_.report();
  }
};
template <typename F1, typename F2>
coupler<F1, F2> couple(F1& f1, F2& f2) {
  return coupler<F1, F2>(f1, f2);
}


struct shredder {
  const int64_t K_;
  std::vector<int32_t> count_;
  shredder(int64_t K) : K_(K), count_(fasta::kmers(K), 0) { }
  void operator()(const std::string& id, const std::string& seq) {
    for (int start = 0; start < (seq.size() - K_ + 1); ++start) {
      std::string kmer = seq.substr(start, K_);
      try {
	int id = fasta::kmer_id(kmer);
	++count_[id];
      } catch (...) {
	std::cout << "illegal kmer = |" << kmer << "|"
		  << "   in ref id = " << id.substr(0, std::min<size_t>(15U, id.size()))
		  << std::endl;
      }
    }
  }
  void report() {
    std::cout << "Writing histogram to histo.csv" << std::endl;
    std::ofstream f("histo.csv");
    f << "id,count" << std::endl;
    for (size_t i = 0; i < count_.size(); ++i)
      f << i << "," << count_[i] << std::endl;
    f.close();
  }
};


struct validator {
  size_t invalid_count_;
  validator() : invalid_count_(0) { }
  void operator()(const std::string& id, const std::string& seq) {
    for (char c : seq) {
      if (!valid_base(c)) {
	++invalid_count_;
	std::cout << "expecting one of {A,C,G,T}, found base = " << c
		  << ";  seq id = " << id
		  << std::endl;
      }
    }
  }
  void report() const {
    std::cout << invalid_count_ << " invalid sequences"
	      << std::endl;
  }
};


struct counter {
  size_t num_targets_;
  size_t num_bases_;
  counter() : num_targets_(0), num_bases_(0) { }
  void operator()(const std::string id, const std::string& seq) {
    ++num_targets_;
    num_bases_ += seq.size();
    if ((num_targets_ % 10000) == 0)
      std::cout << " targets = " << num_targets_ << std::endl;
  }
  void report() const {
    number_format nf;
    std::cout.imbue({std::locale(), &nf});
    std::cout << num_targets_ << " targets" << std::endl;
    std::cout << num_bases_ << " bases" << std::endl;
  }
};


int main() {
  // std::string file = "../data/unpacked/refseq-select-2020-10-22.fasta";
  std::string file = "../data/unpacked/GRCh38_latest_rna.fna";
  std::cout << "fasta file = " << file
            << std::endl;

  auto validate_handler = validator();
  auto count_handler = counter();
  std::size_t K = 10;
  auto shred_handler = shredder(K);

  auto handler = couple(count_handler, shred_handler);
  fasta::parse_file(file, handler);
  handler.report();
  
  return 0;
}
