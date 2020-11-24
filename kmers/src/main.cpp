#include "kmers/fasta.hpp"
#include <string>

int main() {
  std::string file = "../data/unpacked/refseq-select-2020-10-22.fasta";
  seq_map m(10u, file);
  std::cout << std::endl;
  std::cout << m.size() / 1000 << " thousand identifiers"
            << std::endl;
  std::cout << m.total_bases() / 1000000 << " million bases"
            << std::endl;

  std::cout << "Calculating k-mer frequency histogram." << std::endl;
  counter<size_t> c = m.kmer_frequency_counts();
  std::cout << "    Size = " << c.counts_.size() << std::endl;
  for (const auto& freq_count : c.counts_) {
    std::cout << "k-mer frequency = " << freq_count.first
              << ";  count = " << freq_count.second << std::endl;
  }


  // Eigen::VectorXd a(3);  a << 1, 2, 3;
  // Eigen::VectorXd sm_a = softmax(a);

  // std::cout << "K-mer id  = " << kmer_id("") << std::endl;
  // std::cout << "K-mer id A = " << kmer_id("AA") << std::endl;
  // std::cout << "K-mer id AA = " << kmer_id("AA") << std::endl;
  // std::cout << "K-mer id AC = " << kmer_id("AC") << std::endl;
  // std::cout << "K-mer id CA = " << kmer_id("CA") << std::endl;

  // std::cout << "Calculating k-mer by gene matrix" << std::endl;
  // seq_map::mat_t x = m.kmer_gene_matrix();


  // std::cout << "gradient at 0" << std::endl;
  // Eigen::VectorXd y = Eigen::VectorXd::Zero(m.size());
  // log_posterior_gradient lp_g(1, y, x);
  // Eigen::VectorXd grad;
  // Eigen::VectorXd alpha = Eigen::VectorXd::Zero(static_cast<size_t>(m.size()));
  // std::cout << std::endl << "alpha.size() = " << alpha.size() << std::endl;
  // double lp = lp_g(alpha, grad);
  // std::cout << std::endl << "lp = " << lp << std::endl;
}
