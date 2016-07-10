#include "../tests.h"
#include <deal.II/base/tensor.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>

using namespace dealii;


int main()
{
  initlog();
  
    double a[3][3] = {{1, 2, 3}, {3, 4, 5}, {6, 7, 8}};

    const unsigned int dim=3;
    Tensor<2,dim> t(a);
    Vector<double> unrolled(9);
    
    t.unroll(unrolled);

    deallog << "unrolled:";
    for (unsigned i=0; i<9; i++)
      deallog << ' ' << unrolled(i);
    deallog << std::endl;
    
    Assert( std::abs(unrolled.l2_norm() - 14.5945195193) <1e-10,
	    ExcInternalError() );
}
