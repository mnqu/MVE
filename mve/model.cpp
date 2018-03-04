#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <vector>
#include <map>
#include <Eigen/Dense>

#define MAX_STRING 100
#define MAX_VIEW 10

const int hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

struct vocab_node
{
    double degree, degree_view[MAX_VIEW], lambda[MAX_VIEW];
    int num_nb[MAX_VIEW];
    char *name;
};

int node_compare(const void *a, const void *b)
{
    double wei_a = ((struct vocab_node *)a)->degree;
    double wei_b = ((struct vocab_node *)b)->degree;
    if (wei_b > wei_a) return 1;
    else if (wei_b < wei_a) return -1;
    else return 0;
}

struct vocab
{
    vocab_node *node;
    int *vocab_hash;
    int *table;
    long long vocab_max_size, vocab_size, table_size;
    
    vocab()
    {
        node = NULL;
        vocab_hash = NULL;
        table = NULL;
        vocab_max_size = 0;
        vocab_size = 0;
        table_size = 0;
    }
    
    ~vocab()
    {
        if (node != NULL) {free(node); node = NULL;}
        if (vocab_hash != NULL) {free(vocab_hash); vocab_hash = NULL;}
        if (table != NULL) {free(table); table = NULL;}
        vocab_max_size = 0;
        vocab_size = 0;
        table_size = 0;
    }
    
    void init()
    {
        vocab_size = 0;
        vocab_max_size = 1000;
        table_size = 1e8;
        node = (struct vocab_node *)calloc(vocab_max_size, sizeof(struct vocab_node));
        vocab_hash = (int *)calloc(hash_size, sizeof(int));
        for (long long a = 0; a < hash_size; a++) vocab_hash[a] = -1;
    }
    
    int get_hash(char *name)
    {
        unsigned long long a, hash = 0;
        for (a = 0; a < strlen(name); a++) hash = hash * 257 + name[a];
        hash = hash % hash_size;
        return hash;
    }
    
    int add(char *name)
    {
        unsigned int hash, length = strlen(name) + 1;
        if (length > MAX_STRING) length = MAX_STRING;
        node[vocab_size].name = (char *)calloc(length, sizeof(char));
        strcpy(node[vocab_size].name, name);
        node[vocab_size].degree = 0;
        for (int k = 0; k != MAX_VIEW; k++) node[vocab_size].degree_view[k] = 0;
        for (int k = 0; k != MAX_VIEW; k++) node[vocab_size].lambda[k] = 0;
        for (int k = 0; k != MAX_VIEW; k++) node[vocab_size].num_nb[k] = 0;
        vocab_size++;
        // Reallocate memory if needed
        if (vocab_size + 2 >= vocab_max_size) {
            vocab_max_size += 1000;
            node = (struct vocab_node *)realloc(node, vocab_max_size * sizeof(struct vocab_node));
        }
        hash = get_hash(name);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % hash_size;
        vocab_hash[hash] = vocab_size - 1;
        return vocab_size - 1;
    }
    
    int search(char *name)
    {
        unsigned int hash = get_hash(name);
        while (1) {
            if (vocab_hash[hash] == -1) return -1;
            if (!strcmp(name, node[vocab_hash[hash]].name)) return vocab_hash[hash];
            hash = (hash + 1) % hash_size;
        }
        return -1;
    }
    
    void sort()
    {
        int a, size;
        unsigned int hash;
        // Sort the vocabulary and keep </s> at the first position
        qsort(&node[1], vocab_size - 1, sizeof(struct vocab_node), node_compare);
        for (a = 0; a < hash_size; a++) vocab_hash[a] = -1;
        size = vocab_size;
        for (a = 0; a < size; a++) {
            // Hash will be re-computed, as after the sorting it is not actual
            hash = get_hash(node[a].name);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % hash_size;
            vocab_hash[hash] = a;
        }
        node = (struct vocab_node *)realloc(node, (vocab_size + 1) * sizeof(struct vocab_node));
    }
    
    void init_unigram_table()
    {
        int a, i;
        double train_words_pow = 0;
        real d1, power = 0.75;
        table = (int *)malloc(table_size * sizeof(int));
        for (a = 0; a < vocab_size; a++) train_words_pow += pow(node[a].degree, power);
        i = 0;
        d1 = pow(node[i].degree, power) / (real)train_words_pow;
        for (a = 0; a < table_size; a++) {
            table[a] = i;
            if (a / (real)table_size > d1) {
                i++;
                d1 += pow(node[i].degree, power) / (real)train_words_pow;
            }
            if (i >= vocab_size) i = vocab_size - 1;
        }
    }
};

struct sampler
{
    long long n; long long *alias; double *prob;
    
    sampler()
    {
        n = 0; alias = 0; prob = 0;
    }
    
    ~sampler()
    {
        n = 0; if (alias != NULL) { free(alias); alias = NULL; } if (prob != NULL) { free(prob); prob = NULL; }
    }
    
    void init(long long ndata, double *p)
    {
        n = ndata;
        alias = (long long *)malloc(n * sizeof(long long)); prob = (double *)malloc(n * sizeof(double));
        
        long long i, a, g;
        // Local workspace:
        double *P; long long *S, *L;
        P = (double *)malloc(n * sizeof(double)); S = (long long *)malloc(n * sizeof(long long)); L = (long long *)malloc(n * sizeof(long long));
        
        // Normalise given probabilities:
        double sum = 0;
        for (i = 0; i < n; ++i)
        {
            if (p[i] < 0) {fprintf(stderr, "ransampl: invalid probability p[%d]<0\n", (int)(i)); exit(1);}
            sum += p[i];
        }
        if (!sum) {fprintf(stderr, "ransampl: no nonzero probability\n"); exit(1);}
        for (i = 0; i < n; ++i) P[i] = p[i] * n / sum;
        
        // Set separate index lists for small and large probabilities:
        long long nS = 0, nL = 0;
        for (i = n - 1; i >= 0; --i)
        {
            // at variance from Schwarz, we revert the index order
            if (P[i] < 1) S[nS++] = i;
            else L[nL++] = i;
        }
        
        // Work through index lists
        while (nS && nL)
        {
            a = S[--nS]; g = L[--nL]; prob[a] = P[a]; alias[a] = g; P[g] = P[g] + P[a] - 1;
            if (P[g] < 1) S[nS++] = g;
            else L[nL++] = g;
        }
        while (nL) prob[L[--nL]] = 1;
        while (nS) prob[S[--nS]] = 1;
        
        free(P); free(S); free(L);
    }
    
    long long draw(double ran1, double ran2)
    {
        long long i = n * ran1;
        return ran2 < prob[i] ? i : alias[i];
    }
};

struct labeled_node
{
    int node_id;
    std::vector<int> label_ids;
};

struct neuron
{
    BLPVector ac, er;
    void resize(int neuron_size)
    {
        ac.resize(neuron_size);
        er.resize(neuron_size);
    }
    void flush()
    {
        ac.setZero();
        er.setZero();
    }
};

char network_file[MAX_STRING], labeled_file[MAX_STRING], output_file[MAX_STRING];
vocab node;
int binary = 0, num_threads = 1, normalize = 0, vector_size = 100, negative = 5, view_size = 0, depth = 1;
int epoch, epochs = 10;
long long samples = 1, total_samples, edge_count_actual = 0, edge_count_iter = 0;
real lr = 0.025, init_lr, eta = 0.05, phi, init_phi = 0.1;

BLPMatrix view_emb[MAX_VIEW], cont_emb, robust_emb;

// nn is a network between nodes, ln is a network between labels and nodes.
long long num_edges[MAX_VIEW], total_edges = 0;
int **nb_id[MAX_VIEW];
double *hd_wt[MAX_VIEW], **nb_wt[MAX_VIEW];
sampler smp_hd[MAX_VIEW], *smp_nb[MAX_VIEW];

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

// labeled data
std::map<int, int> label2label_id;
std::vector<labeled_node> labeled_data;
int labeled_data_size = 0, label_size = 0;
BLPMatrix Y, W, Z;
neuron neu0, neu1, neu2;

void read_string(char *string, FILE *fi)
{
    int a = 0, ch;
    while (!feof(fi))
    {
        ch = fgetc(fi);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (a > 0)
            {
                if (ch == '\n') ungetc(ch, fi);
                break;
            }
            if (ch == '\n')
            {
                strcpy(string, (char *)"</s>");
                return;
            }
            else continue;
        }
        string[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    string[a] = 0;
}

void learn_vocab()
{
    FILE *fi;
    char string1[MAX_STRING], string2[MAX_STRING], file_name[MAX_STRING];
    double weight;
    long long a, i;
    
    node.init();
    
    for (int view = 0; view != view_size; view++)
    {
        sprintf(file_name, "%s%d", network_file, view);
        fi = fopen(file_name, "rb");
        if (fi == NULL)
        {
            printf("ERROR: training data file not found!\n");
            exit(1);
        }
        num_edges[view] = 0;
        while (1)
        {
            if (fscanf(fi, "%s %s %lf", string1, string2, &weight) != 3) break;
            
            num_edges[view]++;
            total_edges++;
            
            if (total_edges % 10000 == 0)
            {
                printf("%lldK%c", total_edges / 1000, 13);
                fflush(stdout);
            }
            
            i = node.search(string1);
            if (i == -1)
            {
                a = node.add(string1);
                node.node[a].degree = weight;
                node.node[a].degree_view[view] = weight;
                node.node[a].num_nb[view] = 1;
            }
            else
            {
                node.node[i].degree += weight;
                node.node[i].degree_view[view] += weight;
                node.node[i].num_nb[view] += 1;
            }
            
            i = node.search(string2);
            if (i == -1)
            {
                a = node.add(string2);
            }
            //if (i == -1)
            //{
            //    a = node.add(string2);
            //    node.node[a].degree = weight;
            //    node.node[a].degree_view[view] += weight;
            //}
            //else
            //{
            //    node.node[i].degree += weight;
            //    node.node[i].degree_view[view] += weight;
            //}
        }
        fclose(fi);
    }
    
    for (int view = 0; view != view_size; view++)
    {
        double sum = 0;
        for (int k = 0; k != node.vocab_size; k++) sum += node.node[k].degree_view[view];
        for (int k = 0; k != node.vocab_size; k++) node.node[k].degree_view[view] /= sum;
    }
    for (int k = 0; k != node.vocab_size; k++)
    {
        node.node[k].degree = 0;
        for (int view = 0; view != view_size; view++) node.node[k].degree += node.node[k].degree_view[view];
    }
    
    node.sort();
    node.init_unigram_table();
    
    printf("View size: %d\n", view_size);
    printf("Vocab size: %lld\n", node.vocab_size);
    printf("Number of edges: %lld\n", total_edges);
}

void init_vectors()
{
    for (int k = 0; k != view_size; k++) view_emb[k].resize(node.vocab_size, vector_size);
    for (int k = 0; k != view_size; k++) for (int b = 0; b < vector_size; b++) for (int a = 0; a < node.vocab_size; a++)
        view_emb[k](a, b) = 0;
    
    cont_emb.resize(node.vocab_size, vector_size);
    for (int b = 0; b < vector_size; b++) for (int a = 0; a < node.vocab_size; a++)
        cont_emb(a, b) = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    
    robust_emb.resize(node.vocab_size, vector_size);
    robust_emb.setZero();
    
    W.resize(vector_size, label_size);
    for (int b = 0; b < label_size; b++) for (int a = 0; a < vector_size; a++)
        W(a, b) = (rand() / (real)RAND_MAX - 0.5) / (vector_size + label_size);
    
    Z.resize(vector_size, view_size);
    for (int b = 0; b < view_size; b++) for (int a = 0; a < vector_size; a++)
        Z(a, b) = (rand() / (real)RAND_MAX - 0.5) / (vector_size + view_size);
    
    for (int k = 0; k != node.vocab_size; k++) for (int v = 0; v != view_size; v++)
        node.node[k].lambda[v] = (1 + rand()) / (real)RAND_MAX;
    
    neu0.resize(view_size);
    neu1.resize(vector_size);
    neu2.resize(label_size);
}

void read_networks()
{
    FILE *fi;
    char string1[MAX_STRING], string2[MAX_STRING], file_name[MAX_STRING];
    int u, v;
    double w;
    int *pst = (int *)malloc(node.vocab_size * sizeof(int));
    
    for (int view = 0; view != view_size; view++)
    {
        sprintf(file_name, "%s%d", network_file, view);
        
        fi = fopen(file_name, "rb");
        if (fi == NULL)
        {
            printf("ERROR: biterm data file not found!\n");
            exit(1);
        }
        
        for (int k = 0; k != node.vocab_size; k++) pst[k] = 0;
        
        hd_wt[view] = (double *)malloc(node.vocab_size * sizeof(double));
        for (int k = 0; k != node.vocab_size; k++) hd_wt[view][k] = node.node[k].degree_view[view];
        nb_id[view] = (int **)malloc(node.vocab_size * sizeof(int *));
        for (int k = 0; k != node.vocab_size; k++) nb_id[view][k] = (int *)malloc(node.node[k].num_nb[view] * sizeof(int));
        nb_wt[view] = (double **)malloc(node.vocab_size * sizeof(double *));
        for (int k = 0; k != node.vocab_size; k++) nb_wt[view][k] = (double *)malloc(node.node[k].num_nb[view] * sizeof(double));
        
        for (long long k = 0; k != num_edges[view]; k++)
        {
            if (k % 10000 == 0)
            {
                printf("%cRead edges of view %d/%d: %.3lf%%", 13, view + 1, view_size, k / (double)(num_edges[view] + 1) * 100);
                fflush(stdout);
            }
            fscanf(fi, "%s %s %lf", string1, string2, &w);
            u = (int)(node.search(string1));
            v = (int)(node.search(string2));
            nb_id[view][u][pst[u]] = v;
            nb_wt[view][u][pst[u]] = w;
            pst[u] += 1;
        }
        fclose(fi);
        
        smp_hd[view].init(node.vocab_size, hd_wt[view]);
        smp_nb[view] = new sampler [node.vocab_size];
        for (int k = 0; k != node.vocab_size; k++)
        {
            if (node.node[k].num_nb[view] == 0) continue;
            smp_nb[view][k].init(node.node[k].num_nb[view], nb_wt[view][k]);
        }
    }
    printf("\n");
    
    free(pst);
}

void read_labeled_data()
{
    FILE *fi = fopen(labeled_file, "rb");
    int node_id, label, label_id;
    std::vector<int> current_label_ids;
    labeled_node current_labeled_node;
    char string[MAX_STRING];
    
    fi = fopen(labeled_file, "rb");
    while (1)
    {
        if (fscanf(fi, "%s", string) != 1) break;
        
        node_id = node.search(string);
        
        current_label_ids.clear();
        while (1)
        {
            read_string(string, fi);
            if (strcmp(string, "</s>") == 0) break;
            label = atoi(string);
            if (label2label_id[label] == 0) label2label_id[label] = ++label_size;
            label_id = label2label_id[label] - 1;
            current_label_ids.push_back(label_id);
        }
        
        if (node_id != -1)
        {
            labeled_data_size++;
            current_labeled_node.node_id = node_id;
            current_labeled_node.label_ids = current_label_ids;
            labeled_data.push_back(current_labeled_node);
        }
    }
    fclose(fi);
    
    std::random_shuffle(labeled_data.begin(), labeled_data.end());
    
    Y.resize(labeled_data_size, label_size);
    Y.setZero();
    for (int k = 0; k != labeled_data_size; k++)
    {
        int len = (int)(labeled_data[k].label_ids.size());
        for (int i = 0; i != len; i++)
        {
            label_id = labeled_data[k].label_ids[i];
            Y(k, label_id) = 1;
        }
    }
    
    printf("Label size: %d\n", label_size);
    printf("Labeled data size: %d\n", labeled_data_size);
}

void *train_thread(void *id)
{
    int u, v, view, node_id, index;
    long long edge_count = 0, last_edge_count = 0;
    long long tg, lb;
    unsigned long long next_random = (long long)id;
    std::vector<long long> node_list;
    int walk_length = 0;
    real f, g;
    BLPVector error(vector_size);
    
    while (1)
    {
        if (edge_count > samples / num_threads + 2) break;
        
        if (edge_count - last_edge_count > 10000)
        {
            edge_count_actual += edge_count - last_edge_count;
            edge_count_iter += edge_count - last_edge_count;
            last_edge_count = edge_count;
            printf("%cEpoch: %d/%d Learning rate: %f Progress: %.3lf%%", 13,
                   epoch + 1, epochs, lr, (real)edge_count_iter / (real)(samples + 1) * 100);
            fflush(stdout);
            lr = init_lr * (1 - edge_count_actual / (real)(total_samples + 1));
            if (lr < init_lr * 0.0001) lr = init_lr * 0.0001;
        }
        
        view = rand() % view_size;
        
        node_list.clear();
        node_id = smp_hd[view].draw(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        node_list.push_back(node_id);
        walk_length = 0;
        while (1)
        {
            if (node.node[node_id].num_nb[view] == 0) break;
            if (walk_length == depth) break;
            index = smp_nb[view][node_id].draw(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
            node_id = nb_id[view][node_id][index];
            node_list.push_back(node_id);
            walk_length += 1;
        }
        
        for (int k = 1; k <= walk_length; k++)
        {
            u = node_list[0];
            v = node_list[k];
            
            error.setZero();
            
            for (int d = 0; d < negative + 1; d++)
            {
                if (d == 0)
                {
                    tg = v;
                    lb = 1;
                }
                else
                {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    tg = node.table[(next_random >> 16) % node.table_size];
                    lb = 0;
                }
                f = cont_emb.row(tg) * view_emb[view].row(u).transpose();
                f = 1 / (1 + exp(-f));
                g = (lb - f) * lr;
                error += g * cont_emb.row(tg);
                cont_emb.row(tg) += g * view_emb[view].row(u);
            }
            
            view_emb[view].row(u) += error + lr * eta * (robust_emb.row(u) - view_emb[view].row(u)) - 0.0001 * view_emb[view].row(u);
            
            edge_count++;
        }
    }
    pthread_exit(NULL);
}

void normalize_encoding()
{
    if (normalize) for (int a = 0; a < node.vocab_size; a++)
    {
        real len = robust_emb.row(a).norm();
        robust_emb.row(a) /= len;
    }
}

void output()
{
    long long a, b;
    FILE *fo = fopen(output_file, "wb");
    fprintf(fo, "%lld %d\n", node.vocab_size, vector_size);
    for (a = 0; a < node.vocab_size; a++)
    {
        real real_num;
        fprintf(fo, "%s ", node.node[a].name);
        if (binary) for (b = 0; b < vector_size; b++)
        {
            real_num = robust_emb(a, b);
            fwrite(&real_num, sizeof(real), 1, fo);
        }
        else for (b = 0; b < vector_size; b++)
        {
            real_num = robust_emb(a, b);
            fprintf(fo, "%lf ", real_num);
        }
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void softmax(BLPVector &vec)
{
    vec.array() = vec.array().exp();
    real sum = vec.sum();
    vec.array() /= sum;
}

void update_lambda()
{
    BLPVector feature(vector_size), neu(view_size);
    for (int k = 0; k != node.vocab_size; k++)
    {
        feature.setZero();
        for (int v = 0; v != view_size; v++) feature += view_emb[v].row(k);
        feature /= view_size;
        
        neu = feature * Z;
        softmax(neu);
        for (int v = 0; v != view_size; v++) node.node[k].lambda[v] = neu(v);
    }
}

void update_sync()
{
    for (int k = 0; k != node.vocab_size; k++)
    {
        robust_emb.row(k).setZero();
        for (int v = 0; v != view_size; v++) robust_emb.row(k) += view_emb[v].row(k) * node.node[k].lambda[v];
    }
}

void update_weight()
{
    BLPVector error(view_size), feature(vector_size);
    BLPMatrix gW(vector_size, label_size);
    BLPMatrix gZ(vector_size, view_size);
    gW.setZero();
    gZ.setZero();
    for (int k = 0; k != labeled_data_size; k++)
    {
        int node_id = labeled_data[k].node_id;
        
        neu0.flush();
        neu1.flush();
        neu2.flush();
        error.setZero();
        
        // calculate feature
        feature.setZero();
        for (int v = 0; v != view_size; v++) feature += view_emb[v].row(node_id);
        feature /= view_size;
        
        // load neu0
        neu0.ac = feature * Z;
        softmax(neu0.ac);
        
        // neu0 -> neu1
        for (int v = 0; v != view_size; v++) neu1.ac += neu0.ac(v) * view_emb[v].row(node_id);
        
        // neu1 -> neu2
        neu2.ac = neu1.ac * W;
        
        // neu2 activate
        neu2.ac.array() = 1 / ((-neu2.ac.array().array()).exp() + 1);
        
        // loss
        neu2.er = Y.row(k) - neu2.ac;
        
        // neu2 -> neu1
        neu1.er = neu2.er * W.transpose();
        
        // neu1 -> neu0
        for (int v = 0; v != view_size; v++) neu0.er(v) = neu1.er * view_emb[v].row(node_id).transpose();
        
        // neu0 -> error
        for (int v = 0; v != view_size; v++) for (int i = 0; i != view_size; i++)
            error(v) += (neu0.er(v) - neu0.er(i)) * neu0.ac(v) * neu0.ac(i);
        
        // update W and U
        gW += phi * (neu1.ac.transpose() * neu2.er);
        gZ += phi * (feature.transpose() * error);
    }
    
    W += gW / labeled_data_size;
    Z += gZ / labeled_data_size;
}

double compute_error()
{
    BLPVector feature(vector_size);
    double error = 0;
    for (int k = 0; k != labeled_data_size; k++)
    {
        int node_id = labeled_data[k].node_id;
        
        neu0.flush();
        neu1.flush();
        neu2.flush();
        
        // calculate feature
        feature.setZero();
        for (int v = 0; v != view_size; v++) feature += view_emb[v].row(node_id);
        feature /= view_size;
        
        // load neu0
        neu0.ac = feature * Z;
        softmax(neu0.ac);
        
        // neu0 -> neu1
        for (int v = 0; v != view_size; v++) neu1.ac += neu0.ac(v) * view_emb[v].row(node_id);
        
        // neu1 -> neu2
        neu2.ac = neu1.ac * W;
        
        // neu2 activate
        neu2.ac.array() = 1 / ((-neu2.ac.array().array()).exp() + 1);
        
        error += (neu2.ac - Y.row(k)) * (neu2.ac - Y.row(k)).transpose();
    }
    return error / labeled_data_size / label_size;
}

void train()
{
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    init_lr = lr;
    learn_vocab();
    read_networks();
    read_labeled_data();
    init_vectors();
    
    clock_t start = clock();
    printf("Training process:\n");
    total_samples = epochs * samples;
    for (epoch = 0; epoch != epochs; epoch++)
    {
        edge_count_iter = 0;
        edge_count_actual = epoch * samples;
        for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_thread, (void *)a);
        for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        
        update_sync();
        
        phi = init_phi;
        double error = 0, last_error = 1000000000;
        for (int i = 0; i != 100; i++)
        {
            if (phi < 0.0001) break;
            update_weight();
            error = compute_error();
            if (error > last_error) phi /= 2;
            last_error = error;
        }
        printf("error: %lf\n", error);
        update_lambda();
        
        update_sync();
    }
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    normalize_encoding();
    output();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("MVE: Multi-view Network Embedding toolkit.\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-network <string>\n");
        printf("\t\tPrefix of network files for different views.\n");
        printf("\t-views <int>\n");
        printf("\t\tSet the number of views as <int>.\n");
        printf("\t-label <file>\n");
        printf("\t\tUse labeled nodes in <file> for supervision.\n");
        printf("\t-output <file>\n");
        printf("\t\tOutput the learned embeddings into <file>.\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting embeddings in binary moded; default is 0 (off).\n");
        printf("\t-size <int>\n");
        printf("\t\tSet the dimension of node embeddings; default is 100.\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples for negative sampling; default is 5.\n");
        printf("\t-depth <int>\n");
        printf("\t\tThe depth of random walk; default is 1.\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int> Million in an epoch.\n");
        printf("\t-epochs <int>\n");
        printf("\t\tSet the number of training epochs as <int>.\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-lr <float>\n");
        printf("\t\tSet the initial learning rate; default is 0.025\n");
        printf("\t-eta <float>\n");
        printf("\t\tSet the weight of the regularization term as <float>; default is 0.05.\n");
        printf("\t-norm <int>\n");
        printf("\t\tNormalize the learned node embeddings; default is 0 (off).\n");
        printf("\nExamples:\n");
        printf("./mve -network net -views 3 -output vec.txt -binary 1 -size 100 -negative 5 -depth 1 -samples 30 -epochs 20 -threads 20 -norm 1\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-network", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-label", argc, argv)) > 0) strcpy(labeled_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-depth", argc, argv)) > 0) depth = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = atoi(argv[i + 1])*(long long)(1000000);
    if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) lr = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-views", argc, argv)) > 0) view_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-norm", argc, argv)) > 0) normalize = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    train();
    return 0;
}