#ifndef CNN_METRICS_H
#define CNN_METRICS_H

// C++
#include <vector>


class ClassificationEvaluator {
private:
int correct_num = 0; // The number of correct samples currently accumulated
int sample_num = 0; // Current cumulative number of samples
public:
ClassificationEvaluator() = default;
// This batch guessed a few correctly
void compute(const std::vector<int>& predict, const std::vector<int>& labels);
// Check the cumulative accuracy
float get() const;
// Restart statistics
void clear();
};



#endif //CNN_METRICS_H
