#ifndef FASTA_HPP
#define FASTA_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

/**
 * A structure to hold a sparse map from k-mer indexes to k-mer
 * counts with increments.
 */
template <typename T = size_t>
struct counter {
  std::map<T, size_t> counts_;

  /**
   * Increment the count for the specified key by one.
   *
   * @param key key whose count is incremented
   */
  void increment(const T& key) {
    if (counts_.find(key) == counts_.end())
      counts_[key] = 0;
    ++counts_[key];
  }

  /**
   * Return the sum of the values of all of the keys.
   */
  size_t count() const {
    size_t y = 0;
    for (const auto& key_count : counts_)
      y += key_count.second;
    return y;
  }
};


/**
 * Return the identifier in the range 0-3 for the specified base.
 * The mapping is defined alphabetically, indexing from zero, so that
 * 'A' maps to 0, 'C' maps to 1, 'G' maps to 2, and 'T' maps to 3,
 * with any other input throwing an exception.
 *
 * @param c character representing a base
 * @return numerical identifier for the base
 * @throw std::runtime_error if c is not one of 'A', 'C', 'G', or
 * 'T'.
 */
std::size_t base_id(char c) {
  switch (c) {
    case 'A' : return 0;
    case 'C' : return 1;
    case 'G' : return 2;
    case 'T' : return 3;
    default: throw
        std::runtime_error("argument to base_id must be one of:"
                           "'A', 'C', 'G', 'T'");
  }
}

/**
 * Return a numerical identifier for the specified kmer.  Each k-mer
 * is read as a base-4 number given the base_id() for each k-mer.  For
 * a fixed K, this assigns each K-mer gets a unique ID between 0 and
 * 4^K - 1 based on a lexicographic start, indexing from 0.
 *
 * @param kmer string representing a k-mer
 * @return identifier for the k-mer.
 * @throw std::runtime_error if there are characters in kmer other
 * than 'A', 'C', 'G', or 'T'.
 */
std::size_t kmer_id(const std::string& kmer) {
  size_t id = 0;
  for (const char& b : kmer)
    id = 4 * id + base_id(b);
  return id;
}

/**
 * A holder mapping sequence ids to their sequence and k-mers.
 */
struct seq_map {
  using mat_t = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using trip_t = Eigen::Triplet<double>;
  using trips_t = std::vector<trip_t>;

  std::map<std::string, std::string> id_to_seq_;
  int64_t K_;

  /**
   * Return the number of sequences in the domain of the map.
   *
   * @return number of sequences being mapped
   */
  int64_t size() const {
    return id_to_seq_.size();
  }

  /**
   * Return the total number of bases in the range of the map.
   *
   * @return total number of bases in the map
   */
  int64_t total_bases() const {
    return std::accumulate(id_to_seq_.begin(), id_to_seq_.end(), 0,
                           [](const auto& tot, const auto& seq) {
                             return tot + seq.second.size(); });
  }

  /**
   * Return the total number of unique k-mers using this map's k-mer
   *   size.
   *
   * @return total number of unique k-mers
   */
  int64_t kmers() const {
    return pow(4, K_);
  }

  /**
   * Return a vector of strings corresponding to the sequences in the
   * domain of this map.
   *
   * @return the identifiers being mapped
   */
  std::vector<std::string> identifiers() const {
    std::vector<std::string> ids;
    for (const auto& id_seq : id_to_seq_) {
      ids.push_back(id_seq.first);
    }
    return ids;
  }

  /**
   * Return a counter mapping kmers to their frequencies.
   */
  counter<std::string> kmer_frequency() const {
    counter<std::string> c;
    int N = 0;
    for (const auto& id_seq : id_to_seq_) {
      ++N;
      if ((N % 1000) == 0)
	std::cout << N << " / " << id_to_seq_.size() << " isoforms indexed"
		  << std::endl;
      std::string seq = id_seq.second;
      for (size_t i = 0; i < seq.size(); ++i)
	if (seq[i] != 'A' && seq[i] != 'C' && seq[i] != 'G' && seq[i] != 'T')
	  std::cout << "*********** WHOA, found seq[" << i << "] = " << seq[i]
		    << std::endl;
      for (size_t i = 0; i + K_ < seq.size(); ++i) {
        auto kmer = seq.substr(i, K_);
        c.increment(kmer);
      }
    }
    return c;
  }

  /**
   * Return a counter mapping k-mer frequencies to the number of k-mers
   * with that frequency.
   */
  counter<size_t> kmer_frequency_counts() const {
    counter<std::string> c = kmer_frequency();
    counter<size_t> h;
    for (const auto& kmer_freq : c.counts_) {
      h.increment(kmer_freq.second);
    }
    return h;
  }

  mat_t kmer_gene_matrix() const {
    size_t num_kmers = kmers();
    mat_t m(kmers(), size());
    trips_t trips;
    size_t id = 0;
    for (const auto& id_seq : id_to_seq_) {
      std::string seq = id_seq.second;
      counter<size_t> c;
      for (size_t i = 0; i + K_ < seq.size(); ++i) {
        auto kmer = seq.substr(i, K_);
        auto k_id = kmer_id(kmer);
        c.increment(k_id);
      }
      double seq_sz = c.count();
      for (const auto& kmerid_count : c.counts_)
        trips.push_back(trip_t(kmerid_count.first, id,
                               kmerid_count.second / seq_sz));
      ++id;
    }
    m.setFromTriplets(trips.begin(), trips.end());
    return m;
  }

  bool start_seq(std::string& line) {
    return line.size() > 0 && line[0] == '>';
  }
  
  /**
   * Read FASTA formatted data from file with specified name.
   * @param K length of K-mers
   * @param filename name of file from which to read
   */
  seq_map(int64_t K, const std::string& filename)
      : K_(K) {
    std::ifstream in(filename, std::ifstream::in);
    std::string line;
    int64_t count = 0;
    int64_t total_bases = 0;
    while (true) {
      // advance to first readable line
      if (!std::getline(in,line).good()) break;
      if (!start_seq(line)) continue;
      std::string id = line.substr(1);
      std::stringstream val;
      while (std::getline(in, line).good()) {
	if (start_seq(line)) {
	  break;
	}
	val << line;
      }
      std::string bp_seq = val.str();
      id_to_seq_[id] = bp_seq;
      total_bases += bp_seq.size();
    }
    in.close();
    }
  }
};

  
Eigen::VectorXd softmax(const Eigen::VectorXd& alpha) {
  auto alpha_exp = alpha.array().exp();
  return alpha_exp / alpha_exp.sum();
}

struct log_posterior_gradient {
  const double lambda_;
  const Eigen::VectorXd& y_;  // 4^K x 1
  const seq_map::mat_t& x_;   // 4^K x G

  log_posterior_gradient(double lambda,
                         const Eigen::VectorXd& y,
                         const seq_map::mat_t& x)
      : lambda_(lambda), y_(y), x_(x) { }

  double operator()(const Eigen::VectorXd& alpha,
                    Eigen::VectorXd& grad) {
    // implementation lifted and lightly refactored from matrixcalculus.org
    Eigen::VectorXd t0 = alpha.array().exp();
    double t1 = t0.array().sum();
    Eigen::VectorXd t2 = x_.transpose() * (y_.array() * (1 / t1 * x_ * t0).array()).matrix();
    double lpd = y_.transpose() * (x_ * t0 / t1)
        + alpha.dot(alpha) / (2 * lambda_);
    grad = (((1 / t1) * t2).array() * t0.array()).matrix()
        - 1 / (t1 * t1) * t0.transpose() * t2 * t0
        - 1 / lambda_ * alpha;
    return lpd;
  }
};




// (y).dot(np.log(t_1)) - ((a).dot(a) / (2 * s))

#endif
