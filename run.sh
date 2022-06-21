#!/bin/bash
echo ""
echo -e "\033[0;32m Tests de performances disque dur :\033[0m"
echo ""


nbr_thread=1
taille_matrice=1
nbr_repetition=0
while [ $nbr_thread -lt 9 ];
  do
  taille_matrice=60
  while [ $taille_matrice -lt 61 ];
    do
      echo -e "\033[0;32m $nbr_thread $taille_matrice  :\033[0m"
      nbr_repetition=0
        while [ $nbr_repetition -lt 2 ];
          do
              # shellcheck disable=SC2261
              mpiexec -n $nbr_thread ./par $taille_matrice $taille_matrice 0 0 0 0 1

              # shellcheck disable=SC2219
              let nbr_repetition=$nbr_repetition+1
          done
        # shellcheck disable=SC2219
        let taille_matrice=$taille_matrice+1
    done
    # shellcheck disable=SC2219
    let nbr_thread=$nbr_thread+1
  done
