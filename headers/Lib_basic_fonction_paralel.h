/*
||===================================================================||
||--------------------------INIT FUNCTION----------------------------||
||===================================================================||
*/

double *alloc_matrix_p(unsigned long long n,unsigned long long m);
double *write_matrix_p(unsigned long long n,unsigned long long m,unsigned long long file,char *name_file,unsigned long long first_line_value,unsigned long long sparse);
double *matrice_test_p();
void init_vector(double *v,unsigned long long n);
unsigned long long return_n_p(unsigned long long n,unsigned long long world_size,unsigned long long *limite,unsigned long long rank);
double *declare_matrix(unsigned long long line_length,unsigned long long column_length);
double *declare_matrix_column(unsigned long long line_length,unsigned long long column_length);

/*
||===================================================================||
||---------------------Matrice/vector operation----------------------||
||===================================================================||
*/
double *transpose_matrix(double *matrix,unsigned long long n,unsigned long long m);
double *add_vector(double *a,double coef_a,double *b,double coef_b,unsigned long long debut,unsigned long long fin);
double blas1(double *vecteur2,double *vecteur1,unsigned long long n);
double *blas2(double * matrix,unsigned long long n,unsigned long long m,double *v);
double *blas3(double * matrix,unsigned long long n,unsigned long long m,double * matrix_1,unsigned long long n1,unsigned long long m1);
void blas1_div(double *vecteur,unsigned long long n,double div);
void blas2_p(double * matrix,unsigned long long n,unsigned long long m,double *v,double *q);
unsigned long long produit_scalaire(double *vect1,double *vect2,unsigned long long m_v,unsigned long long rank,unsigned long long world_size);

/*
||===================================================================||
||--------------------------Prunsigned long long Function---------------------------||
||===================================================================||
*/
void print_matrix_p(double * matrix,unsigned long long n,unsigned long long m,char *s) ;
void print_vector(double *vect,unsigned long long n,char *s);
void print_value_test(double ** matrix,double *matrix_p,unsigned long long n,unsigned long long m,double *vect_test,double *vect_test_2,unsigned long long print,unsigned long long world_size);
void print_vectors(double *vect1,double *vect2,unsigned long long m_v);

/*
||===================================================================||
||------------------------------Methode------------------------------||
||===================================================================||
*/
double *horner_p(double * matrix,double * x ,unsigned long long limite,unsigned long long n_p,unsigned long long n,unsigned long long m,unsigned long long degre,unsigned long long print,unsigned long long world_size,unsigned long long rank);
double *Classical_Gram_Schmidt(double* x, unsigned long long deg,unsigned long long world_size,unsigned long long rank);
double *Modified_Gram_Schmidt(double* x, unsigned long long deg,unsigned long long world_size,unsigned long long rank);
unsigned long long Arnoldi_Modified_Graham_Schmidt(double* A,double * q, double* h, unsigned long long deg, unsigned long long deg_k,double *init,unsigned long long rank,unsigned long long world_size);

/*
||===================================================================||
||--------------------------Error function---------------------------||
||===================================================================||
*/
void test_error_gs(double * matrix,unsigned long long n,unsigned long long f);
