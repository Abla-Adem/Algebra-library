#include <stdio.h>
#include <mpi.h>
#include<omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Lib_basic_fonction_paralel.h"

/*
||===================================================================||
||--------------------------INIT FUNCTION----------------------------||
||===================================================================||
*/
double *alloc_matrix_p(unsigned long long n,unsigned long long m)
{
    double *matrix= malloc(sizeof (double)*n*m);
    return matrix;
}
double *write_matrix_p(unsigned long long n,unsigned long long m,unsigned long long file,char *name_file,unsigned long long first_line_value,unsigned long long sparse)
{

    double *matrix= malloc(sizeof (double)*n*m);


    if(file==0)
    {

        if(sparse==0)
        {

            for (unsigned long long i = 0; i < n; ++i)
            {
                for (unsigned long long j = 0; j < m; ++j) {
                    //matrix[i][j]=rand()%INT_MAX;
                    matrix[i*m+j]= (rand()%10);

                }
            }
        }
        else
        {
            double random_sparce;
            for (unsigned long long i = 0; i < n; ++i)
            {
                for (unsigned long long j = 0; j < m; ++j) {
                    random_sparce=(double )(rand())/RAND_MAX;
                    //if(random_sparce>0.7)
                    //{

                    matrix[i*m+j]=j+1;


                    //}
                    //else
                    //{
                    //    matrix[i][j]=0;
                    //}
                    //matrix[i][j]=rand()%INT_MAX;

                }
            }
            double *vecteur= malloc(sizeof(double )*m );
            double *result= malloc(sizeof(double )*m );
            for (unsigned long long i = 0; i < m-3; ++i)
            {
                result[i]=0;
                vecteur[i]=i;
                //matrix[i][3+i]=0;
            }

            unsigned long long diagonal=1,diag,indice_lapack_ligne=0;
            double *lapack_matrix= malloc(sizeof (double )*n*(m+1));
            for (unsigned long long i = 0; i < m; ++i)
            {
                lapack_matrix[i]=0;
            }
            for (unsigned long long j = m-1; j > -1; --j) {
                diag=0;
                for (unsigned long long i = 0; i < diagonal; ++i)
                {
                    if(matrix[i*m+j+i]>0.0)
                    {
                        printf("%d %lf",j,matrix[i*m+j+i]);
                        diag=1;
                    }
                }
                if(diag==1)
                {
                    printf("%d \n",diagonal);
                    for (unsigned long long w = 0; w < j; ++w) {
                        lapack_matrix[m*(indice_lapack_ligne+1)+w]=0;
                    }
                    for (unsigned long long z = j; z < m; ++z) {
                        lapack_matrix[m*(indice_lapack_ligne+1)+z]=matrix[(z-j)*m+z];
                    }
                    indice_lapack_ligne=indice_lapack_ligne+1;
                }
                diagonal=diagonal+1;
            }
//            cblas_dgbmv(CblasRowMajor,CblasNoTrans,10,10,0,9,1,lapack_matrix,m,vecteur,1,1,result,1);
            printf("MATRIX %d*%d:\n", n, m);
            for (unsigned long long i = 0; i < n+1; ++i) {
                printf("[");
                for (unsigned long long j = 0; j < m; ++j) {
                    printf(" %lf ,", lapack_matrix[i*m+j]);
                }
                printf("] \n");
            }

            for (unsigned long long j = 0; j < m; ++j) {
                printf("%lf ,",result[j]);
            }

        }
    }
    else
    {
        FILE* mscd = fopen(name_file,"r");
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        unsigned long long car=0,i,j;
        double val;
        for (unsigned long long ii = 0; ii < n; ++ii)
        {
            for (unsigned long long jj = 0; jj < m; ++jj) {
                matrix[ii*m+jj]=0;
            }
        }
        unsigned long long faire=0;
        do {
            car = fgetc(mscd);
            //printf("%d \n",car);
            if(car!=37)
            {

                fseek(mscd, -1, SEEK_CUR);
                fscanf(mscd, "%d %d %lf", &i,&j,&val);
                if(first_line_value==0 && faire==0)
                {
                    faire=1;
                }
                else
                {
                    matrix[i*m+j]=val;
                }

            }
            else
            {
                getline(&line, &len, mscd);

            }
            printf("%d \n",car);

        } while (car != EOF);
    }
    return matrix;
}
double *matrice_test_p()
{
    unsigned long long a=11111111,b = 9090909,c = 10891089,d = 8910891,e = 11108889,f = 9089091,g = 10888911,h = 8909109;
    double *matrice= malloc(sizeof(double *)*8*8);

    for (unsigned long long i = 0; i < 8; ++i) {
        for (unsigned long long j = 0; j < 8; ++j) {
            matrice[i*8+j]=0;
        }
    }
    for (unsigned long long i = 0; i < 8; ++i) {
        matrice[i*8+i]=a;
    }

    unsigned long long pile=0,c_merde=0;
    for (unsigned long long i = 0; i < 8; ++i) {
        matrice[i*8+8-i-1]=-h;

        if(i==3)
        {
            c_merde=1;
        }
        if(i%2==0)
        {
            matrice[i*8+8-i-2]=g;
            matrice[(i+1)*8+8-i-1]=g;
            matrice[i*8+i+1]=-b;
            matrice[(i+1)*8+i]=-b;
            if(pile==0)
            {
                matrice[i*8+i+2]=-c;
                matrice[(i+1)-8+i+3]=-c;

                matrice[i*8+i+3]=d;
                matrice[(i+1)*8+i+2]=d;
                if(c_merde==0)
                {
                    matrice[i*8+i+4]=-e;
                    matrice[(i+1)*8+i+5]=-e;

                    matrice[i*8+i+5]=f;
                    matrice[(i+1)*8+i+4]=f;
                }
                else
                {
                    matrice[i*8+i-4]=-e;
                    matrice[(i+1)*8+i-3]=-e;

                    matrice[i*8+i-3]=f;
                    matrice[(i+1)*8+i-4]=f;
                }

            }
            else
            {
                matrice[i*8+i-2]=-c;
                matrice[(i+1)*8+i-1]=-c;

                matrice[i*8+i-1]=d;
                matrice[(i+1)*8+i-2]=d;
                if(c_merde==0)
                {
                    matrice[i*8+i+4]=-e;
                    matrice[(i+1)*8+i+5]=-e;

                    matrice[i*8+i+5]=f;
                    matrice[(i+1)*8+i+4]=f;
                }
                else
                {
                    matrice[i*8+i-4]=-e;
                    matrice[(i+1)*8+i-3]=-e;

                    matrice[i*8+i-3]=f;
                    matrice[(i+1)+i-4]=f;
                }

            }





            if(pile==0)
            {
                pile=1;
            } else
            {
                pile=0;
            }
        }

    }
    return matrice;

}
unsigned long long return_n_p(unsigned long long n,unsigned long long world_size,unsigned long long *limite,unsigned long long rank)
{
    unsigned long long limite_p;
    unsigned long long n_p;
    if((world_size-1)*(n+( world_size - ( n) % world_size ))/world_size<n)
    {
        n_p=( ( n) + ( world_size - ( n) % world_size ) ) / world_size;
        limite_p=n_p;


        if(rank==(world_size-1))
        {
            limite_p=n - n_p * ( world_size - 1);

            // 81 - 21 * 3 == 18;
        }
        //(81 - 1) / 4 == 20


        // 81 - ( 20 * 3 ) = 21;
        //printf("ici 2 %llu %d %d %d\n",n_p*(world_size),n_p,limite_p,rank);
    }
    else
    {
        if(n%world_size==0)
        {
            n_p=( ( n - 1 ) - ( ( n - 1 ) % world_size ) ) / world_size;
        }
        else {

            n_p = (n - (n % world_size)) / world_size;
            limite_p = n_p;

        }
        if(rank==(world_size-1))
        {
            limite_p=n - n_p * ( world_size - 1);


            // 81 - 21 * 3 == 18;
        }



    }



    if(world_size==0)
    {
        limite_p=n;
        n_p=n;
    }


    *limite=limite_p;
    if(rank==0)
    {
        //printf("ici %llu %d %d %d\n",n_p*(world_size),n_p,*limite,rank);
    }
    return n_p;
}
void init_vector(double *v,unsigned long long n)
{
    for (unsigned long long j = 0; j < n; ++j) {

        v[j]=(rand()/(RAND_MAX+1.))*100;;
        //q[0][j]=j;
    }
}
double *declare_matrix(unsigned long long line_length,unsigned long long column_length)
{
    double *vect1=(double *) malloc(sizeof(double)*line_length*column_length);
    for (unsigned long long i = 0; i < line_length*column_length; ++i) {
        //vect1[i]=rand();
        vect1[i]=i;
    }
    return vect1;
}
double *declare_matrix_column(unsigned long long line_length,unsigned long long column_length)
{
    double *vect1=(double *) malloc(sizeof(double)*line_length*column_length);
    unsigned long long k=0;
    for (unsigned long long i = 0; i < column_length; ++i) {
        //vect1[i]=rand();
        for (unsigned long long j = 0; j < line_length; ++j)
        {
            vect1[i+j*column_length]=k;
            k=k+1;
        }
    }
    return vect1;
}

/*
||===================================================================||
||--------------------------Print Function---------------------------||
||===================================================================||
*/
void print_matrix_p(double * matrix,unsigned long long n,unsigned long long m,char *s) {
    printf("    %s %d*%d:\n", s,n, m);
    for (unsigned long long i = 0; i < n; ++i) {
        printf("    [");
        for (unsigned long long j = 0; j < m; ++j) {
            printf(" %lf ,", matrix[i*m+j]);
        }

        printf("]\n");
    }
}
void print_vector(double *vect,unsigned long long n,char *s)
{
    printf( "%s=[",s);
    for (unsigned long long i = 0; i < n; ++i) {
        printf("%lf ",vect[i]);
    }
    printf("]\n ");
}
void print_value_test(double ** matrix,double *matrix_p,unsigned long long n,unsigned long long m,double *vect_test,double *vect_test_2,unsigned long long print,unsigned long long world_size)
{
    if(print!=0)
    {
        printf("Valeur de test: \n");
        if(print==1)
        {
            print_vector(vect_test,n,"    vect_test");
            print_vector(vect_test_2,n,"   vect_test_2");
        }
        else if(print==2)
        {
            if(world_size>0)
            {
                print_matrix_p(matrix_p,n,m,"matrix");
            }
            else
            {
               // print_matrix(matrix,n,m,"matrix");
            }

            print_vector(vect_test,n,"    vect_test");
        }
        else if (print==3)
        {
            if(world_size==0)
            {
               // print_matrix(matrix,n,m,"matrix");
            }
            else
            {
                print_matrix_p(matrix_p,n,m,"matrix");
            }
            print_vector(vect_test,n,"    x");
        }
        else
        {
            if(world_size==0)
            {
               // print_matrix(matrix,n,m,"matrix");
            }
            else
            {
                print_matrix_p(matrix_p,n,m,"matrix");
            }
        }
    }
}
void print_vectors(double *vect1,double *vect2,unsigned long long m_v)
{
    printf("vector 1 value:[");
    for (unsigned long long i = 0; i < m_v; ++i) {
        printf("%f,",vect1[i]);
    }
    printf("]\n vector 2 value:[");
    for (unsigned long long i = 0; i < m_v; ++i) {
        printf("%f,",vect2[i]);
    }
    printf("]\n");
}

/*
||===================================================================||
||------------------------------Methode------------------------------||
||===================================================================||
*/
double *horner_p(double * matrix,double * x ,unsigned long long limite,unsigned long long n_p,unsigned long long n,unsigned long long m,unsigned long long degre,unsigned long long print,unsigned long long world_size,unsigned long long rank)
{

    double  *sub_matrix= alloc_matrix_p(n_p,m);
    double *v= malloc(sizeof (double )*(m));



    MPI_Scatter(matrix,n_p*n,MPI_DOUBLE,sub_matrix,n_p*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(x,m,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double * y_p = calloc(n_p,sizeof(double ) );
    double * y1 = calloc(n_p*world_size,sizeof(double ) );
    y_p=blas2(sub_matrix,limite,m,x);
    y_p= add_vector(y_p,1,x,1,0,n_p);
    MPI_Allgather(y_p,n_p,MPI_DOUBLE,y1,n_p,
                  MPI_DOUBLE,MPI_COMM_WORLD);
    for (unsigned long long i = 1; i < degre; ++i) {
        y_p=blas2(sub_matrix,limite,m,y1);
        MPI_Allgather(y_p,n_p,MPI_DOUBLE,y1,n_p,
                      MPI_DOUBLE,MPI_COMM_WORLD);

    }
    if(print==1 && rank==0)
    {

        // print_matrix_p(sub_matrix,n,m,"test");
        y1= add_vector(y1,1,x,1,0,m);
        print_vector(y1,n," y");
        //print_vector(x,n," x");
    }

    return y1;
}
//
double *Classical_Gram_Schmidt(double* x, unsigned long long deg,unsigned long long world_size,unsigned long long rank){

    double *v_p= malloc(sizeof (double )*deg);
    double *v_p_sum= malloc(sizeof (double )*deg);
    unsigned long long debut,fin;
    double *q=alloc_matrix_p(deg,deg);
    char s[200];
    for(unsigned long long j=0;j<deg;j++)
    {
        if(rank==0)
        {
            for(unsigned long long j1=0;j1<deg;j1++)
            {
                v_p[j1] =x[j*deg+j1];
            }
        }
        else
        {
            for(unsigned long long j1=0;j1<deg;j1++)
            {
                v_p[j1] =0;
            }
        }

        if(j!=0)
        {
            //***********************
            debut=(j/world_size)*rank;
            fin=(j/world_size)*(rank+1);
            if(rank==world_size-1)
            {
                fin=j;
            }
            if(debut==fin && rank!=0)
            {
                for (unsigned long long i = 0; i < deg; ++i) {
                    v_p[i]=0;
                }
            }
            //***********
            for(unsigned long long k=debut;k<fin;k++)
            {
                double scl = 0.0;
                for (unsigned long long s=0;s<deg;s++)
                {
                    scl = scl + q[k*deg+s]*x[j*deg+s];
                }
                for(unsigned long long j1=0;j1<deg;j1++)
                {
                    v_p[j1] = v_p[j1]-scl*q[k*deg+j1];

                }

            }
            MPI_Allreduce(v_p,v_p_sum,deg,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        }
        else
        {
            for(unsigned long long j1=0;j1<deg;j1++)
            {
                v_p_sum[j1] =x[j1];
            }
        }


        for(unsigned long long j1=0;j1<deg;j1++)
        {
            q[j*deg+j1] = v_p_sum[j1]/sqrt(blas1(v_p_sum,v_p_sum,deg));
        }
        //printf("%d %d %d\n",debut,fin,rank);
        //print_matrix_p(q,deg,deg,"Q");


    }
    return q;
}
//

double *Modified_Gram_Schmidt(double* x, unsigned long long deg,unsigned long long world_size,unsigned long long rank){
    //*****************
    unsigned long long n_p;
    if(deg%world_size==0)
    {
        n_p=( ( deg - 1 ) - ( ( deg - 1 ) % world_size ) ) / world_size;
    }
    else
    {

        n_p=( deg - ( deg % world_size ) ) / world_size;
    }



    //(81 - 1) / 4 == 20
    n_p=deg - n_p * ( world_size - 1);
    // 81 - ( 20 * 3 ) = 21;
    unsigned long long limite=n_p;
    if(rank==world_size-1)
    {
        limite=deg - n_p * ( world_size - 1);
        // 81 - 21 * 3 == 18;
    }
    //**************************
    double *v = alloc_matrix_p(n_p*(world_size+3),deg);
    double *v_p=alloc_matrix_p(n_p*2,deg);
    double *q = alloc_matrix_p(deg,deg);

    for(unsigned long long i=0;i<deg;i++){
        for(unsigned long long j=0;j<deg;j++){
            v[i*deg+j] = x[i*deg+j];
        }
    }

    for(unsigned long long j=0;j<deg;j++)
    {

        for(unsigned long long j1=0;j1<deg;j1++)
        {
            q[j*deg+j1] =v[j*deg+j1]/ sqrt(blas1(&v[j*deg],&v[j*deg],deg));
        }
        if((deg - ( j + 1))>world_size) {
            MPI_Bcast(&q[j*deg],deg,MPI_DOUBLE,0,MPI_COMM_WORLD);
            //***************
            n_p = ((deg - (j + 1)) - ((deg - (j + 1)) % world_size)) / world_size;
            //(81 - 1) / 4 == 20
            n_p = (deg - (j + 1)) - n_p * (world_size - 1);
            // 81 - ( 20 * 3 ) = 21;
            limite = n_p;
            if (rank == world_size - 1) {
                limite = (deg - (j + 1)) - n_p * (world_size - 1);
                // 81 - 21 * 3 == 18;
            }

            //**********************
            //MPI_Scatter(&v[(j+1)*deg], n_p * deg, MPI_DOUBLE, v_p, n_p * deg, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            //sprintf(s,"v_p limite=%d rank=%d j=%d ",(deg - (j + 1)),rank,j);

            //print_matrix_p(v_p,deg,deg,s);
            for (unsigned long long k = 0; k < limite; k++) {
                double scl = 0.0;
                for (unsigned long long s = 0; s < deg; s++) {
                    scl = scl + q[j * deg + s] * v_p[k * deg + s];
                }
                for (unsigned long long j1 = 0; j1 < deg; j1++) {
                    v_p[k * deg + j1] = v_p[k * deg + j1] - scl * q[j * deg + j1];
                }
            }
            MPI_Gather(v_p, n_p * deg, MPI_DOUBLE, &v[(j+1)*deg], n_p * deg,
                       MPI_DOUBLE,0, MPI_COMM_WORLD);

        }
        else
        {
            for (unsigned long long k = j+1; k < deg; k++) {
                double scl = 0.0;
                for (unsigned long long s = 0; s < deg; s++) {
                    scl = scl + q[j * deg + s] * v[k * deg + s];
                }
                for (unsigned long long j1 = 0; j1 < deg; j1++) {
                    v[k * deg + j1] = v[k * deg + j1] - scl * q[j * deg + j1];
                }
            }
        }

    }

    free(q);
    free(v);
    free(v_p);
    return NULL;
}
//
unsigned long long Arnoldi_Modified_Graham_Schmidt(double* A,double * q, double* h, unsigned long long deg, unsigned long long deg_k,double *init,unsigned long long rank,unsigned long long world_size){





    // initialisation aléatoire de q[0]
    for (unsigned long long j = 0; j < deg; ++j) {
        //q[0][j]=(rand()/(RAND_MAX+1.))*100;
        q[j]=init[j];

        //q[0][j]=j;
    }

    //*****
    unsigned long long limite;
    unsigned long long n_p=return_n_p(deg,world_size,&limite,rank);
    double *blas1_v= malloc(sizeof (double )*n_p);
    double *blas1_result= malloc(sizeof (double )*n_p*world_size);
    double *sub_a= alloc_matrix_p(n_p,deg);
    MPI_Scatter(A,n_p*deg,MPI_DOUBLE,sub_a,n_p*deg,MPI_DOUBLE,0,MPI_COMM_WORLD);
    //*******


    //****
    double sum_p,sum;
    MPI_Scatter(q,n_p,MPI_DOUBLE,blas1_v,n_p,MPI_DOUBLE,0,MPI_COMM_WORLD);

    sum_p= blas1(blas1_v,blas1_v,limite);
    sum=0;
    MPI_Allreduce(&sum_p,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    double norm = sqrt(sum);

    //****

    //normalisation de q[0]
    for(unsigned long long i=0;i<deg;i++){ // normalise q[0]
        q[i] = q[i]/norm;
    }
    if(rank==0)
    {
        //print_vector(init,deg,"init");
        //print_vector(q,deg,"q[0]");
    }
    unsigned long long cpt=0;
    for(unsigned long long k=1;k<deg_k+1;k++){


        blas2_p(sub_a,limite,deg,&q[(k-1)*deg],blas1_v);
        MPI_Allgather(blas1_v,n_p,MPI_DOUBLE,&q[k*deg],n_p,MPI_DOUBLE,MPI_COMM_WORLD);

        //*(q[k*deg]) =    blas2(A,&q[(k-1)*deg],deg,deg);
        for(unsigned long long j=0;j<k;j++){
            //******************
            h[j*deg+k-1] =blas1(&q[j*deg],&q[k*deg],deg);

            for(unsigned long long i=0;i<deg;i++){
                q[k*deg+i] = q[k*deg+i] - h[j*deg+k-1]*q[j*deg+i];
            }
        }
        h[k*deg+k-1] = sqrt(blas1(&q[k*deg],&q[k*deg],deg));
        if(h[k*deg+k-1]<0.0000000000001)
        {
            return k;
        }
        for(unsigned long long i=0;i<deg;i++){
            q[k*deg+i] = q[k*deg+i]/h[k*deg+k-1];
        }

    }
    return deg_k;
}


/*
||===================================================================||
||---------------------Matrice/vector operation----------------------||
||===================================================================||
*/
double blas1(double *vecteur2,double *vecteur1,unsigned long long n)
{
    double resultat=0;
    for (unsigned long long i = 0; i < n; ++i) {
        resultat=vecteur2[i]*vecteur1[i]+resultat;
    }
    return resultat;
}
double *blas2(double * matrix,unsigned long long n,unsigned long long m,double *v)
{



    //calcul du produit local
    double *result_p=(double*) malloc(sizeof(double)*n),sum;
    for (unsigned long long i = 0; i < n; ++i) {
        sum=0;
        for (unsigned long long j = 0; j < m; ++j) {
            sum=sum+matrix[i*m+j]*v[j];
        }

        result_p[i]=sum;
    }

    return result_p;

}
double *blas3(double * matrix,unsigned long long n,unsigned long long m,double * matrix_1,unsigned long long n1,unsigned long long m1)
{
    double *result= alloc_matrix_p(n,n1);
    for (unsigned long long i = 0; i < n; ++i) {
        for (unsigned long long j = 0; j < n1; ++j) {
            result[i*n1+j]=0;
            for (unsigned long long k = 0; k < m1; ++k) {
                result[i*n1+j]=result[i*n1+j]+matrix[i*m+k]*matrix_1[j*m1+k];
            }
        }
    }
    return result;
}
void blas1_div(double *vecteur,unsigned long long n,double div)
{
    double resultat=0;
    for (unsigned long long i = 0; i < n; ++i) {
        vecteur[i]=vecteur[i]/div;
    }

}
void blas2_p(double * matrix,unsigned long long n,unsigned long long m,double *v,double *q)
{



    //calcul du produit local
    double sum=0;
    for (unsigned long long i = 0; i < n; ++i) {
        sum=0;
        for (unsigned long long j = 0; j < m; ++j) {
            sum=sum+matrix[i*m+j]*v[j];
        }

        q[i]=sum;
    }


}
double *transpose_matrix(double *matrix,unsigned long long n,unsigned long long m)
{
    double *matrix_transpore= alloc_matrix_p(m,n);
    for (unsigned long long i = 0; i < n; ++i)
    {
        for (unsigned long long j = 0; j < m; ++j)
        {
            matrix_transpore[j*n+i]=matrix[i*m+j];
        }

    }
    return matrix_transpore;
}
double *add_vector(double *a,double coef_a,double *b,double coef_b,unsigned long long debut,unsigned long long fin)
{

    double * y = calloc(fin-debut,sizeof(double ) );
    for (unsigned long long i = debut; i < fin; ++i) {
        y[i]=coef_a*a[i]+coef_b*b[i];
    }
    return y;
}
unsigned long long produit_scalaire(double *vect1,double *vect2,unsigned long long m_v,unsigned long long rank,unsigned long long world_size)
{
    //length of process vector
    unsigned long long taille_vect_p;
    //vector of each process
    double *vect1_p=(double*) malloc(sizeof(double )*m_v);
    double *vect2_p=(double*) malloc(sizeof(double )*m_v);
    //send data to other process
    MPI_Scatter(vect1,m_v/world_size,MPI_DOUBLE,vect1_p,m_v/world_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter(vect2,m_v/world_size,MPI_DOUBLE,vect2_p,m_v/world_size,MPI_DOUBLE,0,MPI_COMM_WORLD);

    taille_vect_p=m_v/world_size;
    if(m_v%world_size!=0)
    {

        if(rank==0)
        {
            MPI_Send(&vect1[m_v-m_v%world_size],m_v%world_size,MPI_DOUBLE,world_size-1,0,MPI_COMM_WORLD);
            MPI_Send(&vect2[m_v-m_v%world_size],m_v%world_size,MPI_DOUBLE,world_size-1,0,MPI_COMM_WORLD);
        }
        else if( rank==world_size-1)
        {
            MPI_Recv(&vect1_p[m_v/world_size],m_v%world_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Recv(&vect2_p[m_v/world_size],m_v%world_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            taille_vect_p=taille_vect_p+m_v%world_size;
        }


    }
    //local scaler product
    double sum=0,sum_global;
    for (unsigned long long i = 0; i < taille_vect_p; ++i) {
        sum=sum+vect1_p[i]*vect2_p[i];
    }
    //sum local
    //printf("sum of process %d=%f\n",rank,sum);
    //global scaler product
    MPI_Reduce(&sum,&sum_global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    return sum_global;
}


/*
||===================================================================||
||--------------------------Error function---------------------------||
||===================================================================||
*/
void test_error_gs(double * matrix,unsigned long long n,unsigned long long f) {
    //print_matrix(matrix,n,n,"matrix");
    double *vect_test = malloc(sizeof(double) * 3);
    vect_test[0] = 0;
    vect_test[1] = 10000;
    vect_test[2] = 0;
    unsigned long long cpt = 0;
    double temp;
    for (unsigned long long i = 0; i < n; ++i) {
        // printf("    ");
        for (unsigned long long j = i + 1; j < n; ++j) {
            cpt = cpt + 1;

            temp = fabs(blas1(&matrix[i * n], &matrix[j * n], n));
            // printf("%.3e \n",temp);

            vect_test[0] = vect_test[0] + temp;
            if (temp > vect_test[2]) {
                vect_test[2] = temp;
            }
            if (temp < vect_test[1]) {
                vect_test[1] = temp;
            }


        }


    }

    //printf("%.3e %.3e %.3e\n",vect_test[0],vect_test[1],vect_test[2]);
    vect_test[0] = vect_test[0] / cpt;
    printf("moy=%.3e min=%.3e max=%.3e n=%d\n", vect_test[0], vect_test[1], vect_test[2], n);
}

