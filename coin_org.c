#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//struct to organize each person
struct person{
  char name[100000];
  long long int unit;
  long long int bills;
  long long int tokens;
};

//this merges the person
void merge(struct person* people, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
 
    // create temp arrays 
    struct person* L=(struct person*)malloc(sizeof(struct person)*n1);

    struct person* R=(struct person*)malloc(sizeof(struct person)*n2);
 
    //Copy data to temp arrays L[] and R[] 
    for (i = 0; i < n1; i++)
        L[i] = people[l + i];
    for (j = 0; j < n2; j++)
        R[j] = people[m + 1 + j];
 
    //Merge the temp arrays back into arr[l..r]
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[i].unit <= R[j].unit) {
            people[k] = L[i];
            i++;
        }
        else {
            people[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there
    are any */
    while (i < n1) {
        people[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there
    are any */
    while (j < n2) {
        people[k] = R[j];
        j++;
        k++;
    }
    free(L);
    free(R);
}
 
/* l is for left index and r is right index of the
sub-array of arr to be sorted */
void mergeSort(struct person* people, int l, int r)
{
    if (l < r) {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l + (r - l) / 2;
 
        // Sort first and second halves
        mergeSort(people, l, m);
        mergeSort(people, m + 1, r);
 
        merge(people, l, m, r);
    }
}
 
/* Function to print an array */


 
int main(void) {
  //initialize all necessary variables 
  int t;
  int b;
  int n;
  scanf("%d",&n);
  char s[21];
  long long int c;
  long long int d;
  
  int result;
  long long int max=0;
  
  struct person* people= (struct person*)malloc(sizeof(struct person)*n);
  //char arr[n];
  
  //read in the names and their tokens and bills
  for(int i=0;i<n;i++){
    scanf("%s %lld %lld", s, &c, &d);
    strcpy(people[i].name,s);
    people[i].tokens=c;
    people[i].bills=d;
  }

  //read in the value of each token and bill
  scanf("%d %d",&t,&b);
  for(int i=0;i<n;i++){
    people[i].unit= people[i].tokens*b+people[i].bills*t;
    printf("%s\n",people[i].name);
  }
  //initiate the sort
  mergeSort(people,0,n-1);
  for(int i=n-1;i>=0;i--){
    //prints from greatest to least
    printf("%s\n",people[i].name);
  }
  free(people);
}