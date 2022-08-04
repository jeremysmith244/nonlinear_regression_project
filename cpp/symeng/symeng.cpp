#include <symengine/expression.h>
# include <iostream>
using SymEngine::Expression;
using SymEngine::eval_double;
using SymEngine::integer;

using namespace std;

int main(){
    Expression x("x");
    auto ex = pow(x+sqrt(Expression(2)), 6);
    cout << ex << endl;
    cout << expand(ex) << endl;
    cout << eval_double(ex.subs({
        {x, integer(2)}
        })) << endl;
}