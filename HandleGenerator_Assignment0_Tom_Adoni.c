#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
  //initialize all proper variables
  char c;
  char s[100001];
  int i;
  int x;
  int len;
  char prevc;
  //reads in the number of names
  scanf(" %d", &x);
  //for loop for the amount of names
  for(int j=0;j<=x;j++){
    prevc=' ';
    i=0;
    c=' ';
    /*while loop to check which characters will be 
    added to the handle*/
    while(c!='\n'){
      scanf("%c",&c);
      if(c=='\n'){
        s[i]=prevc;
        i++;
        s[i]='\0';
        break;
      }
      if(prevc==' ' && c!=' '){
        s[i]=c;
        i++;
      }
      if(c==' '){
        s[i]=prevc;
        i++;
      }
      prevc=c;
    }
    //prints the handle
    printf("%s\n",s);
    //resets the handle for the next name
    s[0]='\0';
  }
  
  return 0;
}