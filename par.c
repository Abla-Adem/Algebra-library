#include <stdio.h>
#include <mpi.h>
#include<omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Lib_basic_fonction_paralel.h"
int main(int argc, char *argv[]) {
    int rank,world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int file,first_line_value,sparse,print;
    double *matrix,*matrix_inv,*q_0,*q_1,*h_acgs,*q_acgs,*h_amgs,*q_amgs,*v,*matrix_colonne;
    double *vect_resultat,resultat,*vect_test,*vect_test_2,*init,*q,*R,*test,*tran_a;
    double *sub_matrix,*sub_vector,*sub_vector_2,resultat_p;
    unsigned long long n,m,k,limite;
    n=strtol(argv[1], NULL, 10);
    m=strtol(argv[2], NULL, 10);
    file=strtol(argv[3], NULL, 10)*500;
    first_line_value=strtol(argv[5], NULL, 10);
    sparse=strtol(argv[6], NULL, 10);
    print=strtol(argv[7], NULL, 10);
    srand(time(NULL));

    vect_test= malloc(sizeof (double )*n);
    vect_test_2= malloc(sizeof (double )*n);
    matrix=alloc_matrix_p(n,m);

    if(rank==0)
    {
        matrix= write_matrix_p(n,m,file,argv[4],first_line_value,sparse);

        for (unsigned long long i = 0; i < n; ++i) {
            vect_test[i]=i;
            vect_test_2[i]=1;
        }
    }
    int n_p=return_n_p(n,world_size,&limite,rank);

    //double **matrix=matrice_test();


    //print valeur test
    if(rank==0)
    {
        double **matrix_seq;
        //print_value_test(matrix_seq,matrix,n,m,vect_test,vect_test_2,print,world_size);
    }


    //Blas 1
    if(print==1)
    {

        sub_vector= malloc(sizeof (double )*n_p);
        sub_vector_2= malloc(sizeof (double )*n_p);
        MPI_Scatter(vect_test,n_p,MPI_DOUBLE,sub_vector,n_p,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Scatter(vect_test_2,n_p,MPI_DOUBLE,sub_vector_2,n_p,MPI_DOUBLE,0,MPI_COMM_WORLD);
        clock_t begin = clock();
        resultat_p= blas1(sub_vector,sub_vector_2,limite);
        clock_t end = clock();
        MPI_Reduce(&resultat_p,&resultat,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        if(rank==0)
        {
            FILE *plot;
            char ss[2000];

            plot= fopen("Blas1.txt","a");
            sprintf(ss,"%d %d %lf\n",world_size,n,time_spent);
            fprintf(plot,ss);
            fclose(plot);

            printf("********************* \n BLAS 1 :vect_test*vect_test_2=%lf %d\n",resultat,rank);
            printf("********************* \n ");
        }

    }

    //Blas 2
    if(print==2)
    {
        clock_t begin,end;
        double time_spent=0;
        sub_matrix= alloc_matrix_p(n_p,m);
        double *vect_final= malloc(sizeof(double )*n_p*world_size);

        MPI_Scatter(matrix,n_p*m,MPI_DOUBLE,sub_matrix,n_p*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(vect_test,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
        begin= clock();
        vect_resultat= blas2(sub_matrix,limite,m,vect_test);
        end = clock();
        MPI_Gather(vect_resultat, n_p, MPI_DOUBLE, vect_final, n_p ,
                   MPI_DOUBLE,0, MPI_COMM_WORLD);

        time_spent = time_spent+(double)(end - begin) / CLOCKS_PER_SEC;
        if(rank==0)
        {
            //printf("%d %d \n",n_p,limite);
            FILE *plot;
            char ss[2000];

            plot= fopen("Blas2.txt","a");
            sprintf(ss,"%d %d %lf\n",world_size,n,time_spent);
            fprintf(plot,ss);
            fclose(plot);
            //print_vector(vect_final,n,"BLAS 2: A*vect_test_2=");
            //printf("********************* \n ");

        }
        free(matrix);
        free(sub_matrix);
        free(vect_resultat);
        free(vect_final);

    }

    //honer methode
    if(print==3)
    {

        vect_resultat= horner_p(matrix,vect_test,limite,n_p,n,m,2,1,world_size,rank);
        if(rank==0)
            printf("********************* \n ");
    }

    //Classical Gram Schmidt
    if(print==4)
    {
        if(rank!=0)
            matrix= alloc_matrix_p(n,m);
        MPI_Bcast(matrix,n*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        q_0=Classical_Gram_Schmidt(matrix,n,world_size,rank);
        if(rank==0)
        {
            printf("Classical Gram Schmidt \n ");
            test_error_gs(q_0,n,2);
        }
        //save_matrix(matrix,n,n,"matrix cgs");
        //save_matrix(q_0,n,n,"Classic_Gram_Schmidt");
        //test_error_gs(q_0,n,2);
    }


    //Arnoldi
    if(print==6)
    {

        init= malloc(sizeof (double )*n);
        double *AQ,*tran_Q,*tran_H,*QH;
        //
        init= malloc(sizeof (double )*n);
        if(rank==0)
            init_vector(init,n);
        MPI_Bcast(init,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

        k=n-1;
        //
        init_vector(init,n);
        h_amgs= alloc_matrix_p(k+1,k);
        q_amgs = alloc_matrix_p(k+1,n);

        int cpt_1=Arnoldi_Modified_Graham_Schmidt(matrix,q_amgs,h_amgs,n,k,init,rank,world_size);
        if(rank==0)
            printf("Arnoldi modified Graham Schmidt %d\n ",cpt_1);






    }
    if(print==7)
    {
        if(rank==0)
        {
            matrix_colonne=declare_matrix_column(n,m);
            matrix=declare_matrix(n,m);
            print_matrix_p(matrix_colonne,n,m,"test");
            test=alloc_matrix_p(n,m);
        }
        else
        {
            matrix_colonne=alloc_matrix_p(n,m);
        }
        MPI_Bcast(matrix_colonne,n*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        sub_matrix= alloc_matrix_p(n/world_size,m);
        MPI_Scatter(matrix,(n/world_size)*m,MPI_DOUBLE,sub_matrix,(n/world_size)*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        double *result_blas3=blas3(sub_matrix,n/world_size,m,matrix_colonne,n,m);
        MPI_Gather(result_blas3, (n/world_size)*m, MPI_DOUBLE, test, (n/world_size)*m ,
                   MPI_DOUBLE,0, MPI_COMM_WORLD);
        if(rank==0)
            print_matrix_p(test,n,m,"result");
    }
    MPI_Finalize();
    return 0;
}

