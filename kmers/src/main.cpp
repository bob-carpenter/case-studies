#include "kmers/dirichlet-sampler.hpp"
#include "kmers/fasta-parser.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <vector>

uint64_t num_kmers(uint64_t K) {
  uint64_t y = 1;
  for (int k = 0; k < K; ++k) {
    y *= 4;
  }
  return y;
}

template <typename T>
void write_val(std::ostream& out, const T& x) {
  out.write((char *)(&x), sizeof(T));
}

template <typename T>
void write_array(std::ostream& out, const T* x, int N) {
  for (int n = 0; n < N; ++n)
    write_val<T>(out, x[n]);
}


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
  return std::all_of(seq.begin(), seq.end(),
                     [](char x) { return valid_base(x); });
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
template <typename F>
F couple(F& f) {
  return f;
}

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
	std::cout << "     # illegal kmer = |" << kmer << "|"
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
      std::cout << "     # targets = " << num_targets_ << std::endl;
  }
  void report() const {
    number_format nf;
    std::cout.imbue({std::locale(), &nf});
    std::cout << "counter.report():  " << num_targets_ << " targets"
	      << std::endl;
    std::cout << "counter.report():  " << num_bases_ << " bases"
	      << std::endl;
  }
};

struct triplet_counter {
  int K_;
  int ref_id_;
  std::vector<Eigen::Triplet<float, int>> kmer_count_;
  triplet_counter(int K, size_t reserve_size)
    : K_(K), ref_id_(0), kmer_count_() {
    kmer_count_.reserve(reserve_size);
  }
  void operator()(const std::string& id, const std::string& seq) {
    int num_kmers = seq.size() - K_ + 1;
    float prob_kmer = 1.0f / num_kmers;
    for (int start = 0; start < num_kmers; ++start) {
      std::string kmer = seq.substr(start, K_);
      try {
	int kmer_id = fasta::kmer_id(kmer);
        kmer_count_.emplace_back(kmer_id, ref_id_, prob_kmer);
      } catch (...) {
	std::cout << "     # illegal kmer = |" << kmer << "|"
		  << "   in ref id = " << id.substr(0, std::min<size_t>(15U, id.size()))
		  << std::endl;
      }
    }
    ++ref_id_;
  }
  void write_matrix(const std::string& filename) {
    int I = ref_id_;
    int M = static_cast<int>(std::pow(4, K_));
    Eigen::SparseMatrix<float, Eigen::RowMajor> x(M, I);
    x.setFromTriplets(kmer_count_.begin(), kmer_count_.end());
    x.makeCompressed();
    std::fstream out(filename, std::ios::binary | std::ios::out);
    if (!out) throw std::runtime_error("cannot open file = " + filename
                                       + " for writing");
    write_val<int>(out, x.rows());
    write_val<int>(out, x.cols());
    write_val<int>(out, x.nonZeros());
    int* outerIndices = x.outerIndexPtr();
    write_array<int>(out, outerIndices, x.rows() + 1);
    int* innerIndices = x.innerIndexPtr();
    write_array<int>(out, innerIndices, x.nonZeros());
    float* values = x.valuePtr();
    write_array<float>(out, values, x.nonZeros());
    out.close();
  }
  void report() const {
    // writing kmer_count_.size(), K_, or ref_id_ causes abnormal termination
    std::cout << "triplet_counter.report():  nothing to report"
              << std::endl;
  }
};


int main() {
  std::string file = "data/unpacked/GRCh38_latest_rna.fna";
  std::cout << "main:  fasta file = " << file
            << std::endl;

  std::size_t K = 10;
  std::cout << "main:  K = " << K
	    << std::endl;
  uint64_t M = num_kmers(K);
  std::cout << "main:  num kmers = " << M
	    << std::endl;

  validator validate_handler = validator();
  counter count_handler = counter();
  // shredder shred_handler = shredder(K);

  int reserve_size = 300000000;
  triplet_counter triplet_handler = triplet_counter(K, reserve_size);

  coupler<counter, triplet_counter> handler = couple(count_handler, triplet_handler);
  fasta::parse_file(file, handler);
  handler.report();
  triplet_handler.write_matrix("xt.bin");

  std::cout << "main:  FINI." << std::endl;
  return 0;
}
