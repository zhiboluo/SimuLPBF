
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/lac/solver_bicgstab.h>

//#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_values.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <map>
#include <sys/stat.h>

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
//	#if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
//		using namespace dealii::LinearAlgebraPETSc;
//	#  define USE_PETSC_LA
//	#elif defined(DEAL_II_WITH_TRILINOS)
		using namespace dealii::LinearAlgebraTrilinos;
//	#else
//	#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
//	#endif
}

#include "rapidxml.hpp"
#include "MaterialParameter.cpp"
#include "updateOperator.cpp"
#include "EquationData.cpp"
//#include "mechanical.cpp"

namespace Evaluation
{
	using namespace std;
	using namespace dealii;
	using namespace Parameters;

//	using namespace Thermo_Elasto_Plastic_Space;
//	class PointHistory<dim>;

	template <int dim>
	class PointValuesEvaluation
	{
	public:
      PointValuesEvaluation (const Point<dim>  &evaluation_point);

      void compute (const hp::DoFHandler<dim>  &dof_handler,
                    const Vector<double>   &solution,
                    Vector<double>         &point_values);

      DeclException1 (ExcEvaluationPointNotFound,
                      Point<dim>,
                      << "The evaluation point " << arg1
                      << " was not found among the vertices of the present grid.");
	private:
      const Point<dim>  evaluation_point;
	};

	template <int dim>
	PointValuesEvaluation<dim>::
	PointValuesEvaluation (const Point<dim>  &evaluation_point)
      :
      evaluation_point (evaluation_point)
    {}

	template <int dim>
    void
    PointValuesEvaluation<dim>::
    compute (const hp::DoFHandler<dim>  &dof_handler,
             const Vector<double>   &solution,
             Vector<double>         &point_values)
	{
//		const unsigned int dofs_per_vertex = dof_handler.get_fe().max_dofs_per_vertex();   //dof_handler.get_fe().dofs_per_vertex;
		const unsigned int dofs_per_vertex = dof_handler.get_fe_collection()[0].dofs_per_vertex;
//		std::cout<<"dofs_per_vertex = "<<dofs_per_vertex<<std::endl;

		AssertThrow (point_values.size() == dofs_per_vertex,
                   ExcDimensionMismatch (point_values.size(), dofs_per_vertex));
		point_values = 1e20;

		typename hp::DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();
		bool evaluation_point_found = false;
		for (; (cell!=endc) && !evaluation_point_found; ++cell)
		{
			const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
			if (dofs_per_cell != 0)
			{
				if (cell->is_locally_owned() && !evaluation_point_found)
					for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex)
					{
						if (cell->vertex(vertex).distance (evaluation_point)
								<
								cell->diameter() * 1e-1)
						{
							for (unsigned int id=0; id!=dofs_per_vertex; ++id)
							{
								point_values[id] = solution(cell->vertex_dof_index(vertex, id, 0));
							}
							evaluation_point_found = true;
							break;
						}
					}
			}
		}

//		AssertThrow (evaluation_point_found,
//                   ExcEvaluationPointNotFound(evaluation_point));
	}

//	template <int dim>
//    void
//    PointValuesEvaluation<dim>::
//	compute_vm_stress (const hp::DoFHandler<dim>  &dof_handler,
//             const Vector<double>   &solution,
//             double         &point_values)
//	{
////		const unsigned int dofs_per_vertex = dof_handler.get_fe().max_dofs_per_vertex();   //dof_handler.get_fe().dofs_per_vertex;
//		const unsigned int dofs_per_vertex = dof_handler.get_fe_collection()[0].dofs_per_vertex;
////		std::cout<<"dofs_per_vertex = "<<dofs_per_vertex<<std::endl;
//
////		AssertThrow (point_values.size() == dofs_per_vertex,
////                   ExcDimensionMismatch (point_values.size(), dofs_per_vertex));
//		point_values = 1e20;
//
//		typename hp::DoFHandler<dim>::active_cell_iterator
//			cell = dof_handler.begin_active(),
//			endc = dof_handler.end();
//		bool evaluation_point_found = false;
//		for (; (cell!=endc) && !evaluation_point_found; ++cell)
//		{
//			const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
//			if (dofs_per_cell != 0)
//			{
//				if (cell->is_locally_owned() && !evaluation_point_found)
//					for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex)
//					{
//						if (cell->vertex(vertex).distance (evaluation_point)
//								<
//								cell->diameter() * 1e-1)
//						{
//							PointHistory<dim> *local_quadrature_points_history
//			                  = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
//			                Assert (local_quadrature_points_history >=
//			                        &quadrature_point_history.front(),
//			                        ExcInternalError());
//			                Assert (local_quadrature_points_history <
//			                        &quadrature_point_history.back(),
//			                        ExcInternalError());
//
//			                // Then loop over the quadrature points of this cell:
//			                double min_distance = 1e10;
//			                unsigned int min_q_index = 0;
//			                for (unsigned int q=0; q<quadrature_formula.size(); ++q)
//			                {
//			                	double tmp_distance = cell->vertex(vertex).distance (local_quadrature_points_history[q].point);
//			                	if (tmp_distance < min_distance)
//			                	{
//			                		min_distance = tmp_distance;
//			                		min_q_index = q;
//			                	}
//			                }
//
////		                	stress_at_qpoint = local_quadrature_points_history[min_q_index].pre_stress;
//		                	double VM_stress = Evaluation::get_von_Mises_stress(local_quadrature_points_history[min_q_index].pre_stress);
//		                	point_values = VM_stress;
//
////							for (unsigned int id=0; id!=dofs_per_vertex; ++id)
////							{
////								point_values[id] = solution(cell->vertex_dof_index(vertex, id, 0));
////							}
//			                evaluation_point_found = true;
//							break;
//						}
//					}
//			}
//		}
//
////		AssertThrow (evaluation_point_found,
////                   ExcEvaluationPointNotFound(evaluation_point));
//	}

 	template <int dim>
 	void print_mesh_info(const Triangulation<dim> &triangulation,
 	                     const std::string        &filename)
 	{
 	  std::cout << "Mesh info:" << std::endl
 	            << " dimension: " << dim << std::endl
 	            << " no. of cells: " << triangulation.n_active_cells() << std::endl;
 	  {
 	    std::map<types::boundary_id, unsigned int> boundary_count;
 	    for (auto cell : triangulation.active_cell_iterators())
 	      {
 	        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
 	          {
 	            if (cell->face(face)->at_boundary())
 	              boundary_count[cell->face(face)->boundary_id()]++;
 	          }
 	      }
 	    std::cout << " boundary indicators: ";
 	    for (const std::pair<const types::boundary_id, unsigned int> &pair : boundary_count)
 	      {
 	        std::cout << pair.first << "(" << pair.second << " times) ";
 	      }
 	    std::cout << std::endl;
 	  }
 	  std::ofstream out (filename);
 	  GridOut grid_out;
 	  grid_out.write_vtk (triangulation, out);
 	  std::cout << " written to " << filename
 	            << std::endl
 	            << std::endl;
 	}


  	 template <int dim>
  	 Point<dim> compute_laser_center(const std::string source_type, const double scan_velocity,
  			 const double segment_start_time, const double segment_end_time,
			 const Point<dim> segment_start_point, const Point<dim> segment_end_point, double time, double & laser_orientation)
  	 {
  		 if (time > segment_end_time) // if current time is cooling time, laser center should be stopped in the segment end point, set the time to segment_end_time
  			 time = segment_end_time;
  		 Point<dim> laserCenter;
  		 double theta = acos((segment_end_point[0] - segment_start_point[0])
				  /sqrt(pow(segment_end_point[0] - segment_start_point[0], 2) +
						  pow(segment_end_point[1] - segment_start_point[1], 2) +
						  pow(segment_end_point[2] - segment_start_point[2], 2)));
		  if (segment_end_point[1] - segment_start_point[1] < 0)
			  theta = -theta;
		  laser_orientation = theta;
//std::cout<<"theta: "<<theta<<std::endl;
		  double vx = scan_velocity*cos(theta), vy = scan_velocity*sin(theta);
		  if (source_type == "Point")
		  { // Point heat source
			  laserCenter[0] = vx*(time - segment_start_time) + segment_start_point[0];
			  laserCenter[1] = vy*(time - segment_start_time) + segment_start_point[1];
			  laserCenter[2] = segment_start_point[2];
//			  return_val = I0*exp(-3*(pow(p[1] - vy*(time - segment_start_time) - segment_start_point[1], 2)/pow(b, 2) +
//						  	  	  	  	  	  	  	  	  pow(p[0] - vx*(time - segment_start_time) - segment_start_point[0], 2)/pow(a, 2) +
//														  pow(p[2] /*- segment_start_point[2]*/, 2)/pow(c, 2)));
		  }
//		  else if (source_type == "Line")
//		  { // Line heat source
//			  const double dt = segment_length/scan_velocity;
//			  double time_temp = floor((time - segment_start_time)/dt)*dt;
//			  laserCenter[0] = vx*(time - segment_start_time) - segment_start_point[0];
//			  laserCenter[1] = vy*(time - segment_start_time) - segment_start_point[1];
//			  laserCenter[2] = segment_start_point[2];
//
//			  return_val = I0*exp(-2*pow( - (p[0] - vx*time_temp - segment_start_point[0])*sin(theta) +
//					  	  	  	  	  	  	  	  (p[1] - vy*time_temp - segment_start_point[1])*cos(theta), 2)/pow(w, 2))*
//			  	  	  	  	  	  (- erf(pow(2, 0.5)*((p[0] - vx*(time_temp + dt) - segment_start_point[0])*cos(theta) +
//			  	  	  	  	  			  (p[1] - vy*(time_temp + dt) - segment_start_point[1])*sin(theta))/w)
//			  	  	  	  	  	   + erf(pow(2, 0.5)*((p[0] - vx*time_temp - segment_start_point[0])*cos(theta) +
//						  	  	  	  	  (p[1] - vy*time_temp - segment_start_point[1])*sin(theta))/w));
//		  }

		  return laserCenter;
  	 }

  	 template <int dim>
  	 std::vector<Point<dim>> compute_vertexes_of_bounding_box(const Point<dim> segment_start_point, const Point<dim> segment_end_point,
  			 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 const double width, const double depth)
	 {
  		Point<dim> P1, P2, P3, P4;
  		std::vector<Point<dim>> output_points(4);

  		if (fabs(segment_end_point[0] - segment_start_point[0]) < 1e-6)
  		{// x0 == x1
  			P1[0] = segment_start_point[0] - width; P1[1] = segment_start_point[1]; P1[2] = segment_start_point[2] - depth;
  			P2[0] = segment_start_point[0] + width; P2[1] = segment_start_point[1]; P2[2] = segment_start_point[2] - depth;
  			P3[0] = segment_start_point[0] + width; P3[1] = segment_end_point[1]; P3[2] = segment_start_point[2] - depth;
  			P4[0] = segment_start_point[0] - width; P4[1] = segment_end_point[1]; P4[2] = segment_start_point[2] - depth;
  		}
  		else if (fabs(segment_end_point[1] - segment_start_point[1]) < 1e-6)
  		{// y0 == y1
  			P1[0] = segment_start_point[0]; P1[1] = segment_start_point[1] + width; P1[2] = segment_start_point[2] - depth;
  			P2[0] = segment_start_point[0]; P2[1] = segment_start_point[1] - width; P2[2] = segment_start_point[2] - depth;
  			P3[0] = segment_end_point[0]; P3[1] = segment_start_point[1] - width; P3[2] = segment_start_point[2] - depth;
  			P4[0] = segment_end_point[0]; P4[1] = segment_start_point[1] + width; P4[2] = segment_start_point[2] - depth;
  		}
  		else
  		{
  			double k = (segment_end_point[1] - segment_start_point[1])/(segment_end_point[0] - segment_start_point[0]);
//  			double k_func = segment_end_point[0] - segment_start_point[0] + k*(segment_end_point[1] - segment_start_point[1]);
  			double width_tmp = width/cos(atan(k));
//  			P1[0] = -(k*width)/(k*k+1) + segment_start_point[0]; P1[1] = width/(k*k+1) + segment_start_point[1]; P1[2] = segment_start_point[2] - depth;
//  			P2[0] = (k_func - k*width)/(k*k+1) + segment_start_point[0]; P2[1] = k*(k_func - k*width)/(k*k+1) + segment_start_point[1] + width; P2[2] = segment_start_point[2] - depth;
//  			P3[0] = (k*width)/(k*k+1) + segment_start_point[0]; P3[1] = -width/(k*k+1) + segment_start_point[1]; P3[2] = segment_start_point[2] - depth;
//  			P4[0] = (k_func + k*width)/(k*k+1) + segment_start_point[0]; P4[1] = k*(k_func + k*width)/(k*k+1) + segment_start_point[1] - width; P4[2] = segment_start_point[2] - depth;

  			P1[0] = -(k*width_tmp)/(k*k+1) + segment_start_point[0]; P1[1] = width_tmp/(k*k+1) + segment_start_point[1]; P1[2] = segment_start_point[2] - depth;
  			P2[0] = (k*width_tmp)/(k*k+1) + segment_start_point[0]; P2[1] = -width_tmp/(k*k+1) + segment_start_point[1]; P2[2] = segment_start_point[2] - depth;
  			P3[0] = (k*k*segment_start_point[0] + k*(segment_end_point[1] - segment_start_point[1]) + k*width_tmp + segment_end_point[0])/(k*k + 1);
  			P3[1] = k*(P3[0] - segment_start_point[0]) + segment_start_point[1] -width_tmp; P3[2] = segment_start_point[2] - depth;
  			P4[0] = (k*k*segment_start_point[0] + k*(segment_end_point[1] - segment_start_point[1]) - k*width_tmp + segment_end_point[0])/(k*k + 1);
  			P4[1] = k*(P4[0] - segment_start_point[0]) + segment_start_point[1] +width_tmp; P4[2] = segment_start_point[2] - depth;

  		}

  		output_points[0] = P1; output_points[1] = P2; output_points[2] = P3; output_points[3] = P4;
  		return output_points;
	 }

  	 template <int dim>
  	 double Get_cross(const Point<dim> p1, const Point<dim> p2, const Point<dim> p)
  	 {
  		 return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1]);
  	 }
  	 template <int dim>
  	 bool Is_Point_in_rectangle(const std::vector<Point<dim>> Points_of_rectangle,  const Point<dim> test_Point)
  	 {
  		 Point<dim> p1 = Points_of_rectangle[0];
  		 Point<dim> p2 = Points_of_rectangle[1];
  		 Point<dim> p3 = Points_of_rectangle[2];
  		 Point<dim> p4 = Points_of_rectangle[3];

  		 return Get_cross(p1, p2, test_Point) * Get_cross(p3, p4, test_Point) >= 0 && Get_cross(p2, p3, test_Point) * Get_cross(p4, p1, test_Point) >= 0;
  	 }

  	 int seperate(unsigned n , int value[])/*该函数将n 分解为2的幂次的和，将非零结果存在数组value中，并返回非0数的个数*/
  	 {
  		 unsigned int i,j=0;
  		 int y;
  		 int temp[sizeof(unsigned int)*8];
  		 for(i=0;i<sizeof(unsigned int)*8;++i)
  		 {
  			 y=n;
  			 y>>=1+i;
  			 y<<=1+i;
  			 temp[i]=n&(~y);
  			 n=y;
  		 }
  		 for(i=0;i<sizeof(unsigned int)*8;++i)
  		 {
  			 if(temp[i]==0)
  				 ;
  			 else
  				 value[j++]=temp[i];
  		 }
  		 return j;
  	 }

  	 int test_main()
  	 {
  		 unsigned a;
  		 int i,n;
  		 int value[sizeof(unsigned)*8];
//  		 printf("Please input the integer:\n");
//  		 scanf("%u",&a);
  		 for(a = 1; a < 35; a++)
  		 {
  			 n=seperate(a,value);

//  			 printf("%d=%d",a,value[0]);
  			printf("%d=%f",a,std::log2(value[0]));
  			 for(i=1;i<n;++i)
//  				 printf("+%d",value[i]);
  				printf("+%f",std::log2(value[i]));
  			 putchar('\n');
  		 }
  		 return 0;
  	 }
//  	 template<int dim>
  	 void determine_refine_level(const unsigned int layer_id, unsigned int &min_level, unsigned int &max_level)
  	 {
//  		 if (layer_id == 1)
//  			 min_level = 4;
//  		 else if(layer_id == 2)
//  			 min_level = 3;
//  		 else if(layer_id == 3)
//  			 min_level = 4;
//  		 else if(layer_id == 4)
//  			 min_level = 2;
//  		 else if(layer_id == 5)
//  			 min_level = 4;
//  		 else if(layer_id == 6)
//  			 min_level = 3;
//  		 else if(layer_id == 7)
//  			 min_level = 4;
//  		 else if(layer_id == 8)
//  			 min_level = 1;
//  		 else if(layer_id == 9)
//  			 min_level = 4;
//  		 else if(layer_id == 10)
//  			 min_level = 3;
//  		 else if(layer_id == 11)
//  			 min_level = 4;
//  		 else if(layer_id == 12)
//  			 min_level = 2;
  		max_level = 5;

  		int value[sizeof(unsigned)*8];
  		seperate(layer_id, value);
		 if (4 - log2(value[0]) < 0)
			 min_level = 0;
		 else
			 min_level = 4 - std::log2(value[0]);
std::cout <<"min_level = "<<min_level<<std::endl;

// 	 for(unsigned int a = 2; a < 105; a++)
//	 {
//		 seperate(a,value);
//		 unsigned int minlevel = 4;
//		 if (4 - log2(value[0]) < 0)
//			 minlevel = 0;
//		 else
//			 minlevel = 4 - log2(value[0]);
//		 std::cout <<"layer_id = "<<a<<", min_level = "<<minlevel<<std::endl;
//	 }

//  		test_main();
  	 }

  	 template<int dim>
  	 double calculate_angle_btw_2_vectors(Point<dim> a, Point<dim> b)
  	 {
  		using namespace std;
  		#define PI 3.1415926
  		/* 向量 A (a,b)  B(c,d) 的夹角为r
  		cosr= 向量A .  向量B / （向量A的摸 * 向量B的摸）
  		*/
//  		double a[2]={1,3},b[2]={3,-1};
  		double ab,a1,b1,cosr;
  		ab=a[0]*b[0]+a[1]*b[1];
  		a1=sqrt(a[0]*a[0]+a[1]*a[1]);
  		b1=sqrt(b[0]*b[0]+b[1]*b[1]);
  		cosr=ab/a1/b1;
//  		if(acos(cosr)*180/PI > 90)
//  			cout<<acos(cosr)*180/PI<<endl;

  		return acos(cosr)*180/PI;
  	 }

  	 template<int dim>
  	 Point<dim> projection_point_to_line(Point<dim> start_Pt, Point<dim> end_Pt, Point<dim> toProject_Pt)
	 {
  		 Point<dim> Projection;
  		 if(fabs(end_Pt[0] - start_Pt[0]) < 1e-6)
  		 {
  			 Projection[0] = start_Pt[0];
  			 Projection[1] = toProject_Pt[1];
  			 Projection[2] = start_Pt[2];
  		 }
  		 else
  		 {
			 double m = (double)(end_Pt[1] - start_Pt[1]) / (end_Pt[0] - start_Pt[0]);
			 double b = (double)start_Pt[1] - (m * start_Pt[0]);

			 Projection[0] = (m * toProject_Pt[1] + toProject_Pt[0] - m * b) / (m * m + 1);
			 Projection[1] = (m * m * toProject_Pt[1] + m * toProject_Pt[0] + b) / (m * m + 1);
			 Projection[2] = start_Pt[2];
  		 }
  		 return Projection;
	 }


 	 template <int dim>
 	 double get_von_Mises_stress(const SymmetricTensor<2, dim> &stress)
 	 {
 		 const double von_Mises_stress = std::sqrt(1.5) * (deviator(stress)).norm();

 		 return von_Mises_stress;
 	 }

 	 template <int dim>
 	 SymmetricTensor<4,dim>
 	 get_stress_strain_tensor (const double lambda, const double mu)
	 {
 		 SymmetricTensor<4,dim> tmp;
 		 for (unsigned int i=0; i<dim; ++i)
 			 for (unsigned int j=0; j<dim; ++j)
 				 for (unsigned int k=0; k<dim; ++k)
 					 for (unsigned int l=0; l<dim; ++l)
 						 tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
 								 	 	 ((i==l) && (j==k) ? mu : 0.0) +
										 ((i==j) && (k==l) ? lambda : 0.0));
 		 return tmp;
	 }

 	 template <int dim>
 	 inline
	 SymmetricTensor<2,dim>
 	 get_strain (const std::vector<Tensor<1,dim> > &grad)
	 {
 		 Assert (grad.size() == dim, ExcInternalError());
 		 SymmetricTensor<2,dim> strain;
 		 for (unsigned int i=0; i<dim; ++i)
 			 strain[i][i] = grad[i][i];

 		 for (unsigned int i=0; i<dim; ++i)
 			 for (unsigned int j=i+1; j<dim; ++j)
 				 strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

 		 return strain;
 	  }



 	 template <int dim>
 	 class ConstitutiveLaw
	 {
	 public:
 		 ConstitutiveLaw (/*MaterialData<dim> material_data*/const AllParameters<dim> &input_data_file);

 		 void set_current_temperature(double current_temperature, unsigned int mat_id);
 		 double get_sigma_y0();
 		 double get_derivative_sigma_y0();
 		 double get_hardening_parameter();
 		 double get_thermal_expansion();
 		 double get_derivative_hardening_parameter();
 		 double get_mu();
 		 double get_K();
 		 void get_elastic_stress_strain_tensor (SymmetricTensor<4, dim> &elastic_stress_strain_tensor);
 		 void get_inverse_elastic_stress_strain_tensor (SymmetricTensor<4, dim> &inverse_elastic_stress_strain_tensor);

 		 bool get_stress_strain_tensor (const SymmetricTensor<2, dim> &strain_tensor,
 	                              	  	  	  	  	  	  	  	  	   SymmetricTensor<4, dim> &stress_strain_tensor);
 	    void get_linearized_stress_strain_tensors (const SymmetricTensor<2, dim> &strain_tensor,
 	                                          SymmetricTensor<4, dim> &stress_strain_tensor_linearized,
 	                                          SymmetricTensor<4, dim> &stress_strain_tensor);
// 		 void get_elastoplasticity_stress_strain_tensor (SymmetricTensor<4, dim> &elastic_plastic_stress_strain_tensor) const;

	 private:
 		 AllParameters<dim> input_data;
 		 MaterialData<dim> mat_data;
 		 double temperature;
 		 unsigned int material_id;
 		 double lambda;
 		 double mu;
 		 double E;	// Young's modulus;
 		 double v; //Poisson's ratio;
 		 double alpha_T; //thermal_expansion;

 		 double       sigma_y0;	// elastic limit. or initial plastic strength
 		 double H;	// (isotropic) hardening parameter
 		 double K;

// 		 const SymmetricTensor<4, dim> elastic_stress_strain_tensor;
// 		 const SymmetricTensor<4, dim> elastic_plastic_stress_strain_tensor;
 		 SymmetricTensor<4, dim> unit_stress_strain_tensor_kappa;
 		 SymmetricTensor<4, dim> unit_stress_strain_tensor_mu;
	 };

 	  template <int dim>
 	  ConstitutiveLaw<dim>::ConstitutiveLaw (/*MaterialData<dim> material_data*/const AllParameters<dim> &input_data_file)
 	    :
	  input_data(input_data_file),
	  mat_data (input_data),
	  temperature (300),
	  material_id(0),
// 	    kappa (E / (3 * (1 - 2 * nu))),
// 	    mu (E / (2 * (1 + nu))),
// 	    sigma_0(sigma_0),
// 	    gamma(gamma),
 	    unit_stress_strain_tensor_kappa (1
 	                                * outer_product(unit_symmetric_tensor<dim>(),
 	                                                unit_symmetric_tensor<dim>())),
 	    unit_stress_strain_tensor_mu (2 * 1
 	                             * (identity_tensor<dim>()
 	                                - outer_product(unit_symmetric_tensor<dim>(),
 	                                                unit_symmetric_tensor<dim>()) / 3.0))
 	  {}

 	 template <int dim>
 	 void
	 ConstitutiveLaw<dim>::set_current_temperature(double current_temperature, unsigned int mat_id)
 	 {
 		 temperature = current_temperature;
 		 material_id = mat_id;
 	 }

 	  template <int dim>
 	  double
 	  ConstitutiveLaw<dim>::get_sigma_y0()
 	  {
 		  sigma_y0 = mat_data.get_elastic_limit(temperature, material_id);
 		  return sigma_y0;
 	  }

 	  template <int dim>
 	  double
 	  ConstitutiveLaw<dim>::get_derivative_sigma_y0()
 	  {
 		  sigma_y0 = mat_data.get_elastic_limit(temperature, material_id);
 		  double sigma_y0_T = mat_data.get_elastic_limit(temperature + 1, material_id);
 		  return sigma_y0_T - sigma_y0;
 	  }

 	  template <int dim>
 	  double
 	  ConstitutiveLaw<dim>::get_hardening_parameter()
 	  {
 		  H = mat_data.get_hardening_parameter(temperature, material_id);
 		  return H;
 	  }

 	  template <int dim>
 	  double
	  ConstitutiveLaw<dim>::get_thermal_expansion()
	  {
 		  alpha_T = mat_data.get_thermal_expansion(temperature, material_id);
 		  return alpha_T;
	  }

 	  template <int dim>
 	  double
 	  ConstitutiveLaw<dim>::get_derivative_hardening_parameter()
 	  {
 		  H = mat_data.get_hardening_parameter(temperature, material_id);
 		  double H_T = mat_data.get_hardening_parameter(temperature + 1, material_id);
 		  return H_T - H;
 	  }

 	  template <int dim>
 	  double
 	  ConstitutiveLaw<dim>::get_mu()
 	  {// shear modulus = E/2(1+v)
 		  mu = mat_data.get_elastic_modulus(temperature, material_id)/(2+2*mat_data.get_poisson_ratio(temperature, material_id));
 		  return mu;
 	  }

 	  template <int dim>
 	  double
 	  ConstitutiveLaw<dim>::get_K()
 	  { // bulk modulus = E/3(1-2v)
 		  K = mat_data.get_elastic_modulus(temperature, material_id)/(3-6*mat_data.get_poisson_ratio(temperature, material_id));
 		  return K;
 	  }

 	  template <int dim>
 	  void
 	  ConstitutiveLaw<dim>::
 	  get_elastic_stress_strain_tensor (SymmetricTensor<4, dim> &stress_strain_tensor)
 	  {
 		  stress_strain_tensor = get_mu()*unit_stress_strain_tensor_mu + get_K()*unit_stress_strain_tensor_kappa;

// 		  double E = mat_data.get_elastic_modulus(temperature, material_id);
// 		  double nu = mat_data.get_poisson_ratio(temperature, material_id);
// 		  mu = E/(2+2*nu);
// 		  double lambda = E*nu/((1+nu)*(1-2*nu));
// 		  SymmetricTensor<4,dim> tmp;
// 		  for (unsigned int i=0; i<dim; ++i)
// 			  for (unsigned int j=0; j<dim; ++j)
// 				  for (unsigned int k=0; k<dim; ++k)
// 					  for (unsigned int l=0; l<dim; ++l)
// 					  {
// 						  tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
// 								  	  	  	  	  	  	  	  	  	  	  	  ((i==l) && (j==k) ? mu : 0.0) +
//																			  ((i==j) && (k==l) ? lambda : 0.0));
// 						  std::cout<<"delta_modulus("<<i<<")("<<j<<")("<<k<<")("<<l<<") = " <<tmp[i][j][k][l] - stress_strain_tensor[i][j][k][l]<< std::endl;
// 					  }


 	  }

 	  template <int dim>
 	  void
 	  ConstitutiveLaw<dim>::
 	  get_inverse_elastic_stress_strain_tensor (SymmetricTensor<4, dim> &inverse_elastic_stress_strain_tensor)
 	  {
 	//	  AssertThrow (determinant(stress_strain_tensor_kappa) != 0, ExcNotImplemented());
 	//	  AssertThrow (determinant(stress_strain_tensor_mu) != 0, ExcNotImplemented());

 		 double mu = get_mu(), kappa = get_K();
 		  inverse_elastic_stress_strain_tensor = invert(kappa*unit_stress_strain_tensor_kappa+mu*unit_stress_strain_tensor_mu);//invert(stress_strain_tensor_kappa) + invert(stress_strain_tensor_mu);
 	  }

 	  template <int dim>
 	  bool
 	  ConstitutiveLaw<dim>::
 	  get_stress_strain_tensor (const SymmetricTensor<2, dim> &strain_tensor,
 	                            SymmetricTensor<4, dim> &stress_strain_tensor)
 	  {
 	    SymmetricTensor<2, dim> stress_tensor;
 	    double mu = get_mu(), kappa = get_K();
 	    double sigma_0 = get_sigma_y0();
 	    double gamma = get_hardening_parameter();

 	    stress_tensor = (kappa*unit_stress_strain_tensor_kappa + mu*unit_stress_strain_tensor_mu)
 	                    * strain_tensor;

 	    const double von_Mises_stress = Evaluation::get_von_Mises_stress(stress_tensor);

 	    stress_strain_tensor = mu*unit_stress_strain_tensor_mu;
 	    if (von_Mises_stress > sigma_0)
 	      {
 	        const double beta = sigma_0 / von_Mises_stress;
 	        stress_strain_tensor *= (gamma + (1 - gamma) * beta);
 	      }

 	    stress_strain_tensor += kappa*unit_stress_strain_tensor_kappa;

 	    return (von_Mises_stress > sigma_0);
 	  }


 	  template <int dim>
 	  void
 	  ConstitutiveLaw<dim>::
 	  get_linearized_stress_strain_tensors (const SymmetricTensor<2, dim> &strain_tensor,
 	                                        SymmetricTensor<4, dim> &stress_strain_tensor_linearized,
 	                                        SymmetricTensor<4, dim> &stress_strain_tensor)
 	  {
 	 	    SymmetricTensor<2, dim> stress_tensor;
 	 	    double mu = get_mu(), kappa = get_K();
 	 	    double sigma_0 = get_sigma_y0();
 	 	    double gamma = get_hardening_parameter();

// 	    SymmetricTensor<2, dim> stress_tensor;
 	    stress_tensor = (kappa*unit_stress_strain_tensor_kappa + mu*unit_stress_strain_tensor_mu)
 	                    * strain_tensor;

 	    stress_strain_tensor = mu*unit_stress_strain_tensor_mu;
 	    stress_strain_tensor_linearized = mu*unit_stress_strain_tensor_mu;

 	    SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
 	    const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();
 	    const double von_Mises_stress = Evaluation::get_von_Mises_stress(stress_tensor);

 	    if (von_Mises_stress > sigma_0)
 	      {
 	        const double beta = sigma_0 / von_Mises_stress;
 	        stress_strain_tensor *= (gamma + (1 - gamma) * beta);
 	        stress_strain_tensor_linearized *= (gamma + (1 - gamma) * beta);
 	        deviator_stress_tensor /= deviator_stress_tensor_norm;
 	        stress_strain_tensor_linearized -= (1 - gamma) * beta * 2 * mu
 	                                           * outer_product(deviator_stress_tensor,
 	                                                           deviator_stress_tensor);
 	      }

 	    stress_strain_tensor += kappa*unit_stress_strain_tensor_kappa;
 	    stress_strain_tensor_linearized += kappa*unit_stress_strain_tensor_kappa;
 	  }

}

namespace Thermo_Elasto_Plastic_Space
{
	using namespace std;
	using namespace rapidxml;
	using namespace dealii;
	using namespace Parameters;
	using namespace EquationData;
	using namespace Evaluation;
//	using namespace mechanical;

	template <int dim>
	struct PointHistory
	{
		SymmetricTensor<2,dim> old_stress;	// used to save the stress of current time step (t + delta_t)
		SymmetricTensor<2,dim> pre_stress;	// used to save the stress of previous time step (t)

		SymmetricTensor<2,dim> old_strain;	// used to save the strain of current time step (t + delta_t)

		SymmetricTensor<2,dim> old_plastic_strain;	// used to save the plastic strain of current time step (t + delta_t)
		SymmetricTensor<2,dim> pre_plastic_strain;	// used to save the plastic strain of previous time step (t)

		double old_effective_plastic_strain;	// used to save the effective plastic strain of current time step (t + delta_t)
		double pre_effective_plastic_strain;	// used to save the effective plastic strain of previous time step (t)

//		SymmetricTensor<2,dim> strain_at_melting;
//		SymmetricTensor<2,dim> thermal_strain_at_melting;

		Point<dim> point;
	};


  template<int dim>
  class HeatEquation
  {
  public:
    HeatEquation(const char *input_material_file, const char *input_layer_file);
    ~HeatEquation();
    void run();

  private:
    void create_coarse_grid();
    void setup_system();
    void assemble_system(/*const Vector<double> Temp_n_k,*/ Vector<double> & initial_cell_material);
    void assemble_system_test(/*const Vector<double> Temp_n_k,*/ Vector<double> & initial_cell_material);
    void update_assemble_system(const Vector<double> Temp_n_k, const Vector<double> initial_cell_material);
    void update_assemble_system_test(const Vector<double> Temp_n_k, const Vector<double> initial_cell_material);
//    double compute_error_residual (const Vector<double> Temp_n_k);
    void solve_time_step();
    void solve_relaxed_Picard();
    void solve_system();
//    void part_height_measure();
    bool cell_is_in_metal_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
    bool cell_is_in_void_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
    void set_active_fe_indices ();
    void refine_mesh (const int min_grid_level,
    								const int max_grid_level);
    void refine_mesh_in_cooling(const int min_grid_level,
    								const int max_grid_level);
    void refine_mesh_btw_layers(const double part_height_before_activate_next_layer,
    								const int max_refine_level);
    void store_old_vectors();
    void transfer_old_vectors();
    void get_attributes_in_layer_node(const xml_node<> *layer_node);
    void get_attributes_in_track_node(const xml_node<> *track_node);
    void get_attributes_in_segment_node(const xml_node<> *segment_node);
    void output_results();

    MPI_Comm                                  mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    parallel::shared::Triangulation<dim>		triangulation;

    hp :: DoFHandler<dim>	dof_handler;
    hp :: FECollection<dim>	fe_collection;
    hp :: QCollection<dim>	quadrature_collection;
    hp :: QCollection<dim-1>	face_quadrature_collection;
    // for material id transfer btw two meshes
    DoFHandler<dim>      material_dof_handler;
    FE_DGQ<dim> history_fe_material;
    IndexSet                                  material_locally_owned_dofs;
    IndexSet                                  material_locally_relevant_dofs;

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix     constraints;

//    SparsityPattern      sparsity_pattern;
//    DynamicSparsityPattern sparsity_pattern;
    LA::MPI::SparseMatrix system_matrix;

    // Vector to visualize the material type of each cell
    Vector<double> 		cell_material;
    Vector<double> 		old_cell_material;

    ConditionalOStream                        pcout;
    std::vector<types::global_dof_index> local_dofs_per_process;
    unsigned int         n_local_cells;

    // Vector to visualize the FE of each cell
    Vector<double> FE_Type;

    Vector<double>       solution;
    Vector<double>       old_solution;
    std::map< Point<dim>, double> map_old_solution;
    std::map< Point<dim>, Vector<double>> map_old_solution_disp;
    std::map<Point<dim>, double> map_material_id;

    std::vector<FullMatrix<double>> cell_mass_matrix_list;
    std::vector<FullMatrix<double>> cell_laplace_matrix_list;
    bool mesh_changed_flg;
    bool mesh_changed_flg_for_mech;

    LA::MPI::Vector       system_rhs;

    // parameters of machine node
    unsigned int total_layers;
    double velocity;
    double beam_diameter;
    double max_segment_length;
    double min_segment_length;

    // printing parameters of layer node
    unsigned int layer_id;
    unsigned int total_scan_tracks;
    Point<dim> laserCenter;
    double orientation;
    double thickness;
    double hatching_space;
    double part_height;		// height of current layer
    Point<dim> layer_start_point;
    Point<dim> layer_end_point;
    double layer_start_time;
    double layer_end_time;
    double idle_time;
    Point<dim> print_direction;

    // printing parameters of track node
    unsigned int track_id;
    double scan_velocity;
    double time_step_point;
    double time_step_line;

    // printing parameters of segment node
    std::string source_type;
    double segment_length;
    double segment_start_time;
    double segment_end_time;
    Point<dim> segment_start_point;
    Point<dim> segment_end_point;

    int         num_dt_line;// number of time step for each line segment
    double               time;
    double               time_step;
    int         timestep_number;		//origin  unsigned int
    double         theta;

    //material properties
    const double heat_capacity;
    const double heat_conductivity;
    const double convection_coeff;
    const double Tamb;
    const double Tinit;

    Parameters::AllParameters<dim>  parameters;
    const std::string layer_file_name;

    TimerOutput        computing_timer;

    std::string				  output_heat_dir, output_mech_dir;
    TableHandler			  table_results,
									  table_results_2,
									  table_results_3,table_results_4, table_results_5;

    bool thermal_mechanical_flg;

    //*********** member and function for mechanical analysis*********************
  private:
	void mechanical_run(unsigned int timestep_number);//, Vector<double> input_solution_temp);
	void setup_mech_system();
	void compute_dirichlet_constraints();
	void move_mesh (bool distort_flag);
	void setup_quadrature_point_history ();
	void update_quadrature_point_history ();
	void assemble_mech_system ();
	void calculate_stress_strain_for_integration_points(unsigned int displacement_loop_iter, const Vector<double> &old_solution_disp);

	void assemble_newton_system (/*const Vector<double> &incremental_displacement_du, */unsigned int newton_step);
//	void compute_nonlinear_residual (const Vector<double> &linearization_point);
//	void solve_mech_time_step ();
	void solve_newton_system();
//	void compute_load_vector_and_residual_force();
    void store_old_vectors_disp();
    void transfer_old_vectors_disp();
	void output_mech_results();// const;
	SymmetricTensor<2,dim> get_thermal_strain (const  double/*Vector<double>*/ &deltatemp, const double &thermoexpan);

//	void solve_newton();
//	void compute_error();

//	Triangulation<dim>   triangulation;
	const unsigned int fe_degree_disp;
	FESystem<dim>      fe_mech;
	FESystem<dim>      fe_none;
	hp :: FECollection<dim>		fe_collection_disp;
	hp::DoFHandler<dim>			dof_handler_disp;

    IndexSet                                  locally_owned_dofs_disp;
    IndexSet                                  locally_relevant_dofs_disp;

	ConstraintMatrix     constraints_disp;
	ConstraintMatrix   	 constraints_dirichlet_and_hanging_nodes;

    hp :: QCollection<dim>	quadrature_collection_disp;
    hp :: QCollection<dim-1>	face_quadrature_collection_disp;
	std::vector<PointHistory<dim> > quadrature_point_history;

	SparsityPattern      sparsity_pattern_disp;
//	SparseMatrix<double> newton_matrix_disp;
	LA::MPI::SparseMatrix newton_matrix_disp;

	Vector<double>       solution_disp;
	Vector<double>		 incremental_displacement;
//	Vector<double>       newton_rhs_disp;
	LA::MPI::Vector 	 newton_rhs_disp;
	Vector<double> 		  fraction_of_plastic_q_points_per_cell;

//	Vector<double>       solution_temp;
	double 						Temp_ref;
//	unsigned int         timestep_number;
	double					expansion_coefficient;
	double	        	 	lame_lamda_coefficient;
	double        	  		lame_mu_coefficient;
//	static const SymmetricTensor<4,dim> stress_strain_tensor;
	ConstitutiveLaw<dim> constitutive_law;

	bool				  transfer_solution;
	unsigned int				current_refinement_cycle;

//	const double alpha; // alpha-method for nonlinear iteration
	//***************************** mechanical end *************************************
  };

  template<int dim>
  HeatEquation<dim>::HeatEquation (const char *input_material_file, const char *input_layer_file)
    :
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  triangulation(mpi_communicator),
  dof_handler(triangulation),
  material_dof_handler(triangulation),
  history_fe_material (2),
	pcout (std::cout, (this_mpi_process == 0)),
	n_local_cells (numbers::invalid_unsigned_int),
	part_height(0.0),
	num_dt_line(5),
	theta(0.5),
    //material properties
	heat_capacity(480*7900),
	heat_conductivity(20),
	convection_coeff(10),
	Tamb(300),	//500 K
	Tinit(300),
	parameters(input_material_file),
	layer_file_name(input_layer_file),
	computing_timer (mpi_communicator,
		                     pcout,
		                     TimerOutput::summary,
		                     TimerOutput::wall_times),
	table_results(),
	table_results_2(),
	table_results_3(),
	table_results_4(),
	table_results_5(),
	//*********** mechanical initialization ****************
	fe_degree_disp (1),
	fe_mech(FE_Q<dim>(QGaussLobatto<1>(fe_degree_disp+1)), dim),
	//	fe_mech(FE_Q<dim>(fe_degree_disp), dim),
	fe_none(FE_Nothing<dim>(fe_degree_disp), dim),
	dof_handler_disp(triangulation),
	Temp_ref(300),
	constitutive_law (parameters)
//	alpha(0.5)
	{
	    time_step = parameters.line_length /parameters.v/200;
	    time = 0.0;
	    timestep_number = 0;
//	    MaterialData<dim> material_data (parameters);

		  fe_collection.push_back (FE_Q<dim>(1));
		  fe_collection.push_back (FE_Nothing<dim>(1));
		  quadrature_collection.push_back(QGauss<dim>(2));
		  face_quadrature_collection.push_back (QGauss<dim-1>(2));
		  face_quadrature_collection.push_back (QGauss<dim-1>(1));

//*********** mechanical initialization ****************
		  fe_collection_disp.push_back (fe_mech);
		  fe_collection_disp.push_back (fe_none);
		  quadrature_collection_disp.push_back(QGauss<dim>(fe_degree_disp + 1));
		  face_quadrature_collection_disp.push_back (QGauss<dim-1>(fe_degree_disp + 1));
		  face_quadrature_collection_disp.push_back (QGauss<dim-1>(1));

//		  output_heat_dir = "/media/admlubuntu/Storage_6T/zhibo/dealii_output/Results_temperature_64x64_leg/";	// local directory
//		  output_mech_dir = "/media/admlubuntu/Storage_6T/zhibo/dealii_output/Results_mechanical_64x64_leg/";
//		  output_heat_dir = "/Volumes/ZBDATA/dealii_output/thermal-stress-part-level-dropbox/Results_temperature_1.6x20_leg/";
//		  output_mech_dir = "/Volumes/ZBDATA/dealii_output/thermal-stress-part-level-dropbox/Results_mechanical_1.6x20_leg/";
		  output_heat_dir = "/Volumes/MENG_2T/dealii_output/Results_temperature_64x64_leg_time/";
		  output_mech_dir = "/Volumes/MENG_2T/dealii_output/Results_mechanical_64x64_leg/";

		  mkdir(output_heat_dir.c_str(), 0777);
		  mkdir(output_mech_dir.c_str(), 0777);

		  {
			  pcout << "Running with "
				#ifdef USE_PETSC_LA
					  	  << "PETSc"
				#else
						  << "Trilinos"
				#endif
						  << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
			  << " MPI rank(s)..." << std::endl;
		  }

		  pcout << "    Using output directory: '" << output_heat_dir << "', for temperature" << std::endl;
		  pcout << "    Using output directory: '" << output_mech_dir << "', for mechanical" << std::endl;
	}

  template <int dim>
  HeatEquation<dim>::~HeatEquation ()
  {
    dof_handler.clear ();
    dof_handler_disp.clear ();
  }

 template <int dim>
  void HeatEquation<dim>::create_coarse_grid ()
  {
	  double max_cell_length = 0.8e-3; //EquationData::g_width; // for current part is 0.05e-3

	  GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
	  pcout<<" read input gmsh file..."<<std::endl;

	  std::ifstream f("build_domain_thinwall_v2.msh");
	  gridin.read_msh(f);
	  print_mesh_info (triangulation, "grid-3D.vtk");


	  double max_width = std::max(EquationData::g_width, 0.05e-3), max_length = std::max(EquationData::g_length, 0.05e-3);
	  int max_refine_level = std::log2(max_cell_length/0.05e-3);
	  pcout<<"max level: "<<max_refine_level<<std::endl;
	  for (int step=0; step<max_refine_level; ++step)
	  {
		  if (EquationData::g_width/2 > 0.05e-3/pow(2, step))
			  if (EquationData::g_width/2 > 0.05e-3/pow(2, step)*2)
				  max_width = EquationData::g_width/2;
			  else
				  max_width = 0.05e-3/pow(2, step)*2;
		  else
			  max_width = 0.05e-3/pow(2, step);
		  if (EquationData::g_length/2 > 0.05e-3/pow(2, step))
			  if (EquationData::g_length/2 > 0.05e-3/pow(2, step)*2)
				  max_length = EquationData::g_length/2;
			  else
				  max_length = 0.05e-3/pow(2, step)*2;
		  else
			  max_length = 0.05e-3/pow(2, step);
		  for (auto cell: triangulation.active_cell_iterators())
		  {
			  if (cell->level() == max_refine_level)
				  continue;
			  Point<dim> centPt = cell->center();
			  if (fabs(centPt[dim -2]) < max_width &&		// |y| < 0.6*max_width
					  fabs(centPt[dim -3]) < max_length &&	// |x| < 0.6*max_length, try to refine the mesh along the border
					  centPt[dim - 1] > 0)									// z > 0, try to refine the mesh above substrate
				  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
				  {
					  const Point<dim> face_center = cell->face(f)->center();

					  if (fabs(face_center[dim - 1] - 0) < 1e-6 &&		// z =0
							  fabs(face_center[dim -2]) < max_width &&	// |y| < max_width
							  fabs(face_center[dim -3]) < max_length)		// |x| < max_length
					  {
						  cell->set_refine_flag ();
						  break;
					  }
				  }
		  }

		  triangulation.execute_coarsening_and_refinement ();
	  }
//	  triangulation.refine_global (initial_global_refinement);   // the size of the original coarse cell should satisfy the layer thickness requirement

	  print_mesh_info (triangulation, "concentrated_mesh.vtk");

	  //setup the initial old_cell_material
	  old_cell_material.reinit(triangulation.n_active_cells());
	  unsigned int cnt_cells (0);

	  // setup the boundary IDs  for thermal convection
	  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
			  	  endc = triangulation.end();
	  for (; cell!=endc; ++cell)
	  {
		  Point<dim> pt_cell_center = cell->center();
		  // Vector to visualize the old material type of each cell
		  if (pt_cell_center[dim - 1] < 0)
		  {
			  cell->set_material_id(0);	// original baseplate material type is solid (material id = 0)
			  unsigned int material_id = cell->material_id();
			  old_cell_material[cnt_cells] = material_id;
		  }
		  else
		  {
			  cell->set_material_id(2);	// original material type is powder (material id = 2)
			  unsigned int material_id = cell->material_id();
			  old_cell_material[cnt_cells] = material_id;
		  }
		  ++ cnt_cells;
	  }
	  pcout <<" ==========  create_coarse_grid() =========="
			  	     <<" number of triangulation = "<< cnt_cells <<std::endl;
  }

  template<int dim>
  void HeatEquation<dim>::setup_system()
  {
	  TimerOutput::Scope timer_section(computing_timer, "Setup");
	  // DOF distribution using proper finite element
	  dof_handler.distribute_dofs(fe_collection);
	  material_dof_handler.distribute_dofs(history_fe_material);

	  locally_owned_dofs = dof_handler.locally_owned_dofs ();
	  DoFTools::extract_locally_relevant_dofs (dof_handler,
													 locally_relevant_dofs);
	  material_locally_owned_dofs = material_dof_handler.locally_owned_dofs ();
	  DoFTools::extract_locally_relevant_dofs (material_dof_handler,
													 material_locally_relevant_dofs);

	  n_local_cells
	  	  = GridTools::count_cells_with_subdomain_association (triangulation,
	                                                           triangulation.locally_owned_subdomain ());
	  local_dofs_per_process = dof_handler.n_locally_owned_dofs_per_processor();

	  double limit = 0;
	  limit = part_height;

	  cell_mass_matrix_list.resize(triangulation.n_active_cells());
	  cell_laplace_matrix_list.resize(triangulation.n_active_cells());
	  mesh_changed_flg = true;
//	  cell_rhs_list.resize(triangulation.n_active_cells());

	  cell_material.reinit(triangulation.n_active_cells());
	  FE_Type.reinit(triangulation.n_active_cells());
	  unsigned int cnt_cells (0), active_cnt_cells(0);

	  // setup the boundary IDs  for thermal convection
	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  cell = dof_handler.begin_active(),
			  endc = dof_handler.end();
	  for (; cell!=endc; ++cell)
	  {
//		  const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
//		  if (dofs_per_cell != 0 ) // Skip the cells which are in the void domain
		  if (cell->is_locally_owned())
		  {
			  active_cnt_cells++;
			  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//				  if (cell->face(f)->at_boundary())
			  {
				  const Point<dim> face_center = cell->face(f)->center();

				  if (fabs(face_center[dim - 1] - limit) < 1e-6)
				  {	// top layer boudary conditon--- convection & radiation
//						  cout << "heat flux boudary:" <<endl;
					  //cell->face(f)->set_boundary_id (1); //close temperary
				  }
				  else if (cell->face(f)->at_boundary())
				  {// set up other boudary surface at boundary
					  if (fabs(face_center[dim - 1] + EquationData::g_base_height) < 1e-6 )//&& fabs(face_center[0])<10e-3 && fabs(face_center[1])<10e-3)
					  {	// bottom surface, Dirchlet boundary
						  cell->face(f)->set_boundary_id (2);
	//					  cell->face(f)->set_boundary_id (1);
	//							  cout << "Dirchlet boundary:" <<endl;
//						  if (fabs(face_center[0]) > 43.1158e-3)// && fabs(face_center[1]) > 9.6e-3)
//							  cell->face(f)->set_boundary_id (22);//Dirichlet boundary for mech analysis
					  }
					  else if (fabs(face_center[dim - 1]) < 1e-6)
					  {	// top surface of substrate
						  cell->face(f)->set_boundary_id (3);
					  }
					  else
					  {	// thermal convection and radiation boundary, // other boundary faces, set to convection & radiation
						  cell->face(f)->set_boundary_id (4);
					  }
//					  cell->face(f)->set_boundary_id (1);
				  }
			  }
		  }

		  // Vector to visualize the material type of each cell
		  unsigned int material_id = cell->material_id();
		  cell_material[cnt_cells] = material_id;
		  // Vector to visualize the FE of each cell
		  unsigned int fe_index = cell->active_fe_index();
		  FE_Type[cnt_cells] = fe_index;
		  ++ cnt_cells;
	  }
//	  std::cout<<"cnt_cells of dof = "<<cnt_cells <<std::endl;
	  cout<<"active_cnt_cells of dof = "<<active_cnt_cells <<", total active cell: "<<cnt_cells<<"-- in Processer:"<<this_mpi_process<<std::endl;

	  constraints.clear ();
	  constraints.reinit(locally_relevant_dofs);
	  DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
	  VectorTools::interpolate_boundary_values(dof_handler,
		  	  	  	  	  	  	  	  	  	  	  	  	  	  21,
															  EquationData::BoundaryValues<dim>(),
															  constraints); // interpolate Dirichlet boundary condition
	  constraints.close();

	  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
	  sparsity_pattern.reinit(dof_handler.n_dofs(), dof_handler.n_dofs(), locally_relevant_dofs);
	  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern,
                                    constraints,
                                    /*keep_constrained_dofs = */ false); // used to true in sequential computing
	  SparsityTools::distribute_sparsity_pattern (sparsity_pattern,
	    											local_dofs_per_process,
	                                                mpi_communicator,
	                                                locally_relevant_dofs);

	  system_matrix.reinit (locally_owned_dofs,
	                          locally_owned_dofs,
	                          sparsity_pattern,
	                          mpi_communicator);
	  system_rhs.reinit(locally_owned_dofs,mpi_communicator);

//	  {// view the sparse matrix printout
//		  SparsityPattern static_sparsity_pattern;
//		  static_sparsity_pattern.copy_from (sparsity_pattern);
//
//		  std::string filename = "static_sparsity_pattern-"
//										 + Utilities::int_to_string(timestep_number, 3) +
//										 "." +
//										 Utilities::int_to_string
//										 (dof_handler.get_triangulation().locally_owned_subdomain(), 4);
//		  filename = output_heat_dir + filename;
//		  std::ofstream output((filename + ".svg").c_str());
//		  static_sparsity_pattern.print_svg (output);
//	  }
  }


  template <int dim>
  void HeatEquation<dim>::assemble_system_test(/*const Vector<double> Temp_n_k, */Vector<double> & initial_cell_material)
  {
	  TimerOutput::Scope timer_section(computing_timer, "Assembling Test");

	  hp::FEValues<dim> hp_fe_values (fe_collection, quadrature_collection,
			  	  	  	  	  	  update_values | update_gradients |
								  update_quadrature_points | update_JxW_values);
	  // Finite element evaluated in quadrature points of the faces of a cell.
	  hp::FEFaceValues<dim> hp_fe_face_values(fe_collection, face_quadrature_collection,
													update_values | update_quadrature_points |
													update_quadrature_points | update_JxW_values);

	  system_matrix = 0;
	  system_rhs = 0;

	  const unsigned int           n_q_points    = quadrature_collection[0].size();		// n_q_points = 8
	  const unsigned int 			n_face_q_points = face_quadrature_collection[0].size(); // quadrature points on faces = 4
	  const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell; // dofs_per_active_cell = 8

	  FullMatrix<double> 			cell_matrix(dofs_per_active_cell, dofs_per_active_cell),
			  	  	  	  	  	  	  	  	  	  	cell_mass_matrix(dofs_per_active_cell, dofs_per_active_cell),
													cell_laplace_matrix(dofs_per_active_cell, dofs_per_active_cell);
	  FullMatrix<double>			    face_cell_mass_matrix (dofs_per_active_cell, dofs_per_active_cell);
	  Vector<double>	 		    	cell_rhs(dofs_per_active_cell);
	  Vector<double>      	        	face_cell_rhs (dofs_per_active_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_active_cell);

//	  EquationData::InputHeatFlux<dim> input_heat_flux_func(parameters,
//			  source_type, scan_velocity, segment_start_time, segment_length, orientation, segment_start_point, segment_end_point);

	  std::string old_source_type = source_type;
	  if (source_type == "Line" && time - time_step*1.5 < segment_start_time )
	  {// the first time step of current segment. change the previous source type to Point if swith from point to line, or Line if swith from line to point
//		  if (source_type == "Line")
		  {// heat source for previous segment is point
			  old_source_type = "Point";
			  std::cout<<"change to Point"<<std::endl;
		  }
//		  else
//		  {// heat source for previous segment is Line
//			  old_source_type = "Line";
//			  std::cout<<"change to Line"<<std::endl;
//		  }
	  }
//	  EquationData::InputHeatFlux<dim> input_heat_flux_func_old(parameters,
//			  old_source_type, scan_velocity, segment_start_time, segment_length, orientation, segment_start_point, segment_end_point);


	  EquationData::InputHeatFlux<dim> rhs_function(parameters,
			  source_type, scan_velocity, segment_start_time, segment_end_time, segment_length, orientation, segment_start_point, segment_end_point);
//	  EquationData::RightHandSide<dim> rhs_function;
	  std::vector<double>  rhs_old_values (n_q_points), rhs_values (n_q_points);

//	  std::vector<double> 		cell_old_solu_vectors(n_q_points);//, cell_solu_vectors(n_q_points);
	  Vector<double> 			cell_old_solu_Vectors(n_q_points);//, cell_solu_Vectors(n_q_points);

	  double     specific_heat_old, specific_heat_Tnk,
			  	  	  conductivity_old, conductivity_Tnk,
					  convectivity_old, convectivity_Tnk,
					  emissivity_old, emissivity_Tnk;

	  // obtain the center point of laser beam and scaning orientation, only execute once. Later on, it will be needed in the stage of refine_mesh
	  laserCenter = compute_laser_center(source_type, scan_velocity, segment_start_time, segment_end_time,
	    												segment_start_point, segment_end_point, time - time_step, orientation); // calculate the center of laser
	  double a = parameters.w, b = 1*parameters.w, c = 1*parameters.w; // three axis radius of the ellipsoid

	  MaterialData<dim> material_data (parameters);
	  double melt_point = parameters.melt_point;// = 1650 + 273;
	  double density = parameters.density;
	  const double Stephan_Boltzmann = 5.67e-8;
	  unsigned int cnt_cells (0), active_cnt_cells(0);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  cell = dof_handler.begin_active(),
					  endc = dof_handler.end();
	  for (; cell!=endc; ++cell, ++cnt_cells)
	  {
		  if (cell->is_locally_owned())
		  {
		  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
		  if (dofs_per_cell != 0)
		  {
			  active_cnt_cells++;
			  cell_matrix = 0;
			  cell_mass_matrix = 0;
			  cell_laplace_matrix = 0;
			  cell_rhs = 0;

			  hp_fe_values.reinit(cell);
			  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

			  rhs_function.set_time(time);
			  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);	// Qn ---> rhs_values
			  rhs_function.set_time(time - time_step);
			  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_old_values);		// Qn-1 ---> rhs_old_values

//			  cell_old_solu_vectors.resize(n_q_points);

//			  Vector<double> 			cell_old_solu_Vectors_tmp(n_q_points);
			  cell->get_dof_values(old_solution, cell_old_solu_Vectors); // get the local dof values at each node
//			  fe_values.get_function_values(old_solution, cell_old_solu_vectors);		// Tn-1  ---> cell_old_solu_vectors
//			  fe_values.get_function_values(Temp_n_k, cell_solu_vectors);	// Tnk  ---> cell_solu_vectors

			  double avg_old_tmp = 0., avg_tmp = 0.;
			  double num_melt_point_old = 0;//, num_melt_point = 0;

			  for (unsigned int t = 0; t < n_q_points; t++)
			  {
//				  cell_old_solu_Vectors[t] = cell_old_solu_vectors[t];
				  avg_old_tmp += cell_old_solu_Vectors[t];
				  if (cell_old_solu_Vectors[t] >= melt_point)
					  num_melt_point_old++;
//				  cell_solu_Vectors[t] = cell_solu_vectors[t];
//				  avg_tmp += cell_solu_vectors[t];
//				  if (cell_solu_vectors[t] >= melt_point)
//					  num_melt_point++;
//				  if(!(fabs(cell_old_solu_vectors[t] - 300) < 1e-3))
//				  {
//					  pcout<<"dof values"<<"("<<t<<")="<<cell_old_solu_Vectors_tmp[t] <<endl;
//					  pcout<<"fe  values"<<"("<<t<<")="<<cell_old_solu_vectors[t] <<endl;
//				  }

			  }
//			  avg_tmp /= n_q_points;

			  avg_old_tmp /= n_q_points;
			  avg_tmp = avg_old_tmp;
//			  num_melt_point /= n_q_points;
			  num_melt_point_old /= n_q_points;
//			  num_melt_point = num_melt_point_old;

			  unsigned int old_material_id = old_cell_material[cnt_cells];
			  specific_heat_old = material_data.get_specific_heat (avg_old_tmp, old_material_id);
			  conductivity_old = material_data.get_conductivity (avg_old_tmp, old_material_id);
			  convectivity_old = material_data.get_convectivity (avg_old_tmp, old_material_id);
			  emissivity_old = material_data.get_emissivity (avg_old_tmp, old_material_id);

//
//			  if (old_material_id == 2)
//			  {// material id of last iteration (or old_material_id?) is powder
//				  {
//					  if (num_melt_point > 5/8)//(avg_tmp >= melt_point)
//						  cell->set_material_id(1);
//					  else
//						  if (cell->material_id() != 2)
//							  cell->set_material_id(2); //(0);
//				  }
//			  }
//			  else if (old_material_id == 1)
//			  {// material id of last iteration is liquid
//				  if (num_melt_point < 0.5)//(avg_tmp < melt_point)
//					  cell->set_material_id(0);
//				  else
//					  cell->set_material_id(1);
//			  }
//			  else if (old_material_id == 0)
//			  {// material id of last iteration is solid
//				  {
//					  if (num_melt_point > 4/8)//(avg_tmp >= melt_point)
//						  cell->set_material_id(1);
//					  else
//						  cell->set_material_id(0);
//				  }
//			  }

		    	Point<dim> centerPt = cell->center();
//		    	if (centerPt[2] <= limit)
		    	{// only consider the cell under the part height (activated cell)
		    		double local_x = (centerPt[0] - laserCenter[0])*cos(orientation) + (centerPt[1] - laserCenter[1])*sin(orientation),
		    					 local_y = -(centerPt[0] - laserCenter[0])*sin(orientation) + (centerPt[1] - laserCenter[1])*cos(orientation);
		    		if (local_x > 0) // point in local axis lie in back of the laser heat. In this circumstance, elongate the long radius of the ellipsoid
		    			a = 0.25*parameters.w;
		    		else
		    			a = parameters.w;
		    	    double distance = pow(local_x/(1*a), 2) + pow(local_y/(1*b), 2) +
		    	    							  pow((centerPt[2] - laserCenter[2])/(1*c), 2) - 1;
		    		if (distance <= 0)
		    		{// center point of current active cell fall into the defined surface
		    				cell->set_material_id(0); // set material id to solid()
		    		}
		    	}
		    	if (cell->material_id() == 1)
		    		cell->set_material_id(0);
		    	unsigned int material_id = cell->material_id();
		    	if(old_material_id == material_id)
		    	{
		    		specific_heat_Tnk = specific_heat_old;
		    		conductivity_Tnk = conductivity_old;
		    		convectivity_Tnk = convectivity_old;
		    		emissivity_Tnk = emissivity_old;
		    	}
		    	else
		    	{
		    		cell_material[cnt_cells] = material_id;
		    		initial_cell_material[cnt_cells] = material_id;
		    		specific_heat_Tnk = material_data.get_specific_heat (avg_tmp, material_id);
		    		conductivity_Tnk = material_data.get_conductivity (avg_tmp, material_id);
		    		convectivity_Tnk = material_data.get_convectivity (avg_tmp, material_id);
		    		emissivity_Tnk = material_data.get_emissivity (avg_tmp, material_id);
		    	}

		    	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		    	{
		    		for (unsigned int i=0; i<dofs_per_cell; ++i)
		    		{
		    			cell_rhs(i) += (rhs_values [q_point]*theta + rhs_old_values [q_point]*(1-theta))*
								  	  	  	  	  time_step*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point);	// cell_rhs = Tau_n*theta*Fn+Tau_n*(1-theta)*Fn-1
		    		}
		    	}
//		    	cell_rhs *= (rhs_values [q_point]*theta + rhs_old_values [q_point]*(1-theta));

		    	if(mesh_changed_flg) //(timestep_number == 1)
		    	{
					  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
					  {
						  for (unsigned int i=0; i<dofs_per_cell; ++i)
						  {
							  for (unsigned int j=0; j<dofs_per_cell; ++j)
							  {
								  cell_mass_matrix(i, j) += (fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point))*fe_values.JxW(q_point);
								  cell_laplace_matrix(i, j) += (fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point))*fe_values.JxW(q_point);
							  }
						  }
					  }
					  cell_mass_matrix_list[cnt_cells] = cell_mass_matrix;
					  cell_laplace_matrix_list[cnt_cells] = cell_laplace_matrix;
		//			  cell_rhs_list[cnt_cells] = cell_rhs;
		    	}
		    	else
		    	{
		    		cell_mass_matrix = cell_mass_matrix_list[cnt_cells];
		    		cell_laplace_matrix = cell_laplace_matrix_list[cnt_cells];
//		    		cell_rhs = cell_rhs_list[cnt_cells];
		    	}

			  double coeff_cell_mass_matrix = density*specific_heat_Tnk,
					  	  coeff_cell_laplace_matrix = conductivity_Tnk*time_step*theta;
			  cell_matrix.add(coeff_cell_mass_matrix, cell_mass_matrix, coeff_cell_laplace_matrix, cell_laplace_matrix);
			  	  // cell_matrix = M(Tn) + Tau_n*theta*K(Tn)

//			  double coeff_cell_mass_matrix_old = density*specific_heat_old;
			  double coeff_cell_laplace_matrix_old = -time_step*(1-theta)*conductivity_old;
			  Vector<double> temp_mass(n_q_points);
			  Vector<double> temp_laplace(n_q_points);
			  cell_mass_matrix.vmult(temp_mass, cell_old_solu_Vectors);
			  cell_laplace_matrix.vmult(temp_laplace, cell_old_solu_Vectors);
	          cell_rhs.add(coeff_cell_mass_matrix, temp_mass, coeff_cell_laplace_matrix_old, temp_laplace);
	          	  // cell_rhs += M(Tn)*Tn-1 - Tau_n*(1-theta)*K(Tn-1)*Tn-1

//	          Point<dim> centerPt = cell->center();
	          if (layer_id > 5)		//(centerPt[dim - 1] > thickness*(layer_id - 1))
	          {
	        	  // Apply Boundary conditions
	        	  //
	        	  //
	        	  face_cell_mass_matrix = 0;
	        	  face_cell_rhs = 0;

				  double coeff_face_cell_rhs_conv = time_step*Tamb*(theta*convectivity_Tnk + (1-theta)*convectivity_old);
				  double coeff_face_cell_rhs_rad =time_step*Tamb*Stephan_Boltzmann*(theta*emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
																						  +
																						  (1-theta)*emissivity_old*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
				  double coeff_face_cell_rhs = coeff_face_cell_rhs_conv + coeff_face_cell_rhs_rad;

				  double coeff_face_cell_matrix_conv = convectivity_Tnk*time_step*theta;
				  double coeff_face_cell_matrix_rad = emissivity_Tnk*Stephan_Boltzmann*(avg_tmp + Tamb) // avg_tmp should be replaced with Tnk
																						*(avg_tmp*avg_tmp + Tamb*Tamb)*time_step*theta;
				  double coeff_face_cell_matrix = coeff_face_cell_matrix_conv + coeff_face_cell_matrix_rad;

				  double coeff_face_cell_rhs_mass_conv = -time_step*(1-theta)*convectivity_old;
				  double coeff_face_cell_rhs_mass_rad = - time_step*(1-theta)*emissivity_old*Stephan_Boltzmann*(avg_old_tmp + Tamb)
																					  *(avg_old_tmp*avg_old_tmp + Tamb*Tamb);
				  double coeff_face_cell_rhs_mass = coeff_face_cell_rhs_mass_conv + coeff_face_cell_rhs_mass_rad;

				  for (unsigned int face_number=0; face_number <
											  GeometryInfo<dim>::faces_per_cell; ++face_number)
				  {
					  // Tests to select the cell faces which belong to the convection and radiation boundary
					  if (cell->face(face_number)->at_boundary()
									  && (cell->face(face_number)->boundary_id() == 1 ))
					  {// convection boundary & radiation boundary
						  // Term to be added in the RHS
						  hp_fe_face_values.reinit (cell, face_number);
						  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();

						  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
						  {
							  // Computation and storage of the value of the shape functions on boundary face integration points
							  for (unsigned int i=0; i<dofs_per_cell; ++i)
							  {
								  for (unsigned int j=0; j<dofs_per_cell; ++j)
								  {
									  face_cell_mass_matrix(i, j) += (fe_face_values.shape_value(i, q_point)*fe_face_values.shape_value(j, q_point))*fe_face_values.JxW(q_point);
								  }
								  face_cell_rhs(i) += fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
								  // face_cell_rhs = Tau_n*theta*Tamb*Vconv_n + Tau_n*(1-theta)*Tamb*Vconv_n-1 +
								  //	  						Tau_n*theta*Tamb*Vrad_n + Tau_n*(1-theta)*Tamb*Vrad_n-1
							  }
						  }

						  cell_matrix.add(coeff_face_cell_matrix, face_cell_mass_matrix);

						  Vector<double> temp_mass_convection(dofs_per_cell);
						  face_cell_mass_matrix.vmult(temp_mass_convection, cell_old_solu_Vectors);
						  cell_rhs.add(coeff_face_cell_rhs_mass, temp_mass_convection);
						  cell_rhs.add(coeff_face_cell_rhs, face_cell_rhs);
					  }

//					  // apply the input heat flux boundary condition: ID = 1
//					  if (cell->face(face_number)->at_boundary()
//									  && (cell->face(face_number)->boundary_id() == 1) && time <= layer_end_time)
//					  {// input heat flux boundary
//						  // Term to be added in the RHS
//						  hp_fe_face_values.reinit (cell, face_number);
//						  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();
//
//						  std::vector<double>  input_heat_flux_current_values (n_face_q_points),
//														   input_heat_flux_old_values (n_face_q_points);
//						  input_heat_flux_func.set_time(time);
//						  input_heat_flux_func.value_list (fe_face_values.get_quadrature_points(),
//																			input_heat_flux_current_values);
//
//
//						  input_heat_flux_func_old.set_time(time - time_step);
//						  input_heat_flux_func_old.value_list (fe_face_values.get_quadrature_points(),
//																			input_heat_flux_old_values);
//
//						  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
//						  {
//							  // input heat flux value.
//							  // Computation and storage of the value of the shape functions on boundary face integration points
//							  double coeff_heat_flux_at_q_point = time_step*theta*input_heat_flux_current_values[q_point] +
//																								  time_step*(1 - theta)*input_heat_flux_old_values[q_point];
//							  for (unsigned int i=0; i<dofs_per_cell; ++i)
//							  {
//								  cell_rhs(i) += coeff_heat_flux_at_q_point*fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
//							  }
//						  }
//					  }

				  }

	          }
			  cell->get_dof_indices (local_dof_indices);
//			  for (unsigned int i=0; i<dofs_per_cell; ++i)
//			  {
//				  for (unsigned int j=0; j<dofs_per_cell; ++j)
//					  system_matrix.add (local_dof_indices[i],
//							  	  	  	  	  	  	  local_dof_indices[j],
//													  cell_matrix(i,j));
//				  system_rhs(local_dof_indices[i]) += cell_rhs(i);
//			  }
			  constraints.distribute_local_to_global (cell_matrix, cell_rhs,
	  		                                              local_dof_indices,
	  		                                              system_matrix, system_rhs); // this doesn't work for hp elements, don't know why


		  }
	  	  }
	  }

	  system_matrix.compress(VectorOperation::add);
	  system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void HeatEquation<dim>::assemble_system(/*const Vector<double> Temp_n_k, */Vector<double> & initial_cell_material)
  {
	  TimerOutput::Scope timer_section(computing_timer, "Assembling");

	  hp::FEValues<dim> hp_fe_values (fe_collection, quadrature_collection,
			  	  	  	  	  	  update_values | update_gradients |
								  update_quadrature_points | update_JxW_values);
	  // Finite element evaluated in quadrature points of the faces of a cell.
	  hp::FEFaceValues<dim> hp_fe_face_values(fe_collection, face_quadrature_collection,
													update_values | update_quadrature_points |
													update_quadrature_points | update_JxW_values);

	  system_matrix = 0;
	  system_rhs = 0;

	  const unsigned int           n_q_points    = quadrature_collection[0].size();		// n_q_points = 8
	  const unsigned int 			n_face_q_points = face_quadrature_collection[0].size(); // quadrature points on faces = 4
	  const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell; // dofs_per_active_cell = 8

	  FullMatrix<double> 			cell_matrix(dofs_per_active_cell, dofs_per_active_cell),
			  	  	  	  	  	  	  	  	  	  	cell_mass_matrix(dofs_per_active_cell, dofs_per_active_cell),
													cell_laplace_matrix(dofs_per_active_cell, dofs_per_active_cell);
	  FullMatrix<double>			    face_cell_mass_matrix (dofs_per_active_cell, dofs_per_active_cell);
	  Vector<double>	 		    	cell_rhs(dofs_per_active_cell);
	  Vector<double>      	        	face_cell_rhs (dofs_per_active_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_active_cell);

//	  EquationData::InputHeatFlux<dim> input_heat_flux_func(parameters,
//			  source_type, scan_velocity, segment_start_time, segment_length, orientation, segment_start_point, segment_end_point);

	  std::string old_source_type = source_type;
	  if (source_type == "Line" && time - time_step*1.5 < segment_start_time )
	  {// the first time step of current segment. change the previous source type to Point if swith from point to line, or Line if swith from line to point
//		  if (source_type == "Line")
		  {// heat source for previous segment is point
			  old_source_type = "Point";
			  std::cout<<"change to Point"<<std::endl;
		  }
//		  else
//		  {// heat source for previous segment is Line
//			  old_source_type = "Line";
//			  std::cout<<"change to Line"<<std::endl;
//		  }
	  }
//	  EquationData::InputHeatFlux<dim> input_heat_flux_func_old(parameters,
//			  old_source_type, scan_velocity, segment_start_time, segment_length, orientation, segment_start_point, segment_end_point);


	  EquationData::InputHeatFlux<dim> rhs_function(parameters,
			  source_type, scan_velocity, segment_start_time, segment_end_time, segment_length, orientation, segment_start_point, segment_end_point);
//	  EquationData::RightHandSide<dim> rhs_function;
	  std::vector<double>  rhs_old_values (n_q_points), rhs_values (n_q_points);

//	  std::vector<double> 		cell_old_solu_vectors(n_q_points);//, cell_solu_vectors(n_q_points);
	  Vector<double> 			cell_old_solu_Vectors(n_q_points);//, cell_solu_Vectors(n_q_points);

	  double     specific_heat_old, specific_heat_Tnk,
			  	  	  conductivity_old, conductivity_Tnk,
					  convectivity_old, convectivity_Tnk,
					  emissivity_old, emissivity_Tnk;

	  // obtain the center point of laser beam and scaning orientation, only execute once. Later on, it will be needed in the stage of refine_mesh
	  laserCenter = compute_laser_center(source_type, scan_velocity, segment_start_time, segment_end_time,
	    												segment_start_point, segment_end_point, time - time_step, orientation); // calculate the center of laser
	  double a = parameters.w, b = 1*parameters.w, c = 1*parameters.w; // three axis radius of the ellipsoid

	  MaterialData<dim> material_data (parameters);
	  double melt_point = parameters.melt_point;// = 1650 + 273;
	  double density = parameters.density;
	  const double Stephan_Boltzmann = 5.67e-8;
	  unsigned int cnt_cells (0), active_cnt_cells(0);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  cell = dof_handler.begin_active(),
					  endc = dof_handler.end();
	  for (; cell!=endc; ++cell, ++cnt_cells)
	  {
		  if (cell->is_locally_owned())
		  {
		  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
		  if (dofs_per_cell != 0)
		  {
			  active_cnt_cells++;
			  cell_matrix = 0;
			  cell_mass_matrix = 0;
			  cell_laplace_matrix = 0;
			  cell_rhs = 0;

			  hp_fe_values.reinit(cell);
			  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

			  rhs_function.set_time(time);
			  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);	// Qn ---> rhs_values
			  rhs_function.set_time(time - time_step);
			  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_old_values);		// Qn-1 ---> rhs_old_values

//			  cell_old_solu_vectors.resize(n_q_points);

//			  Vector<double> 			cell_old_solu_Vectors_tmp(n_q_points);
			  cell->get_dof_values(old_solution, cell_old_solu_Vectors); // get the local dof values at each node
//			  fe_values.get_function_values(old_solution, cell_old_solu_vectors);		// Tn-1  ---> cell_old_solu_vectors
//			  fe_values.get_function_values(Temp_n_k, cell_solu_vectors);	// Tnk  ---> cell_solu_vectors

			  double avg_old_tmp = 0., avg_tmp = 0.;
			  double num_melt_point_old = 0, num_melt_point = 0;

			  for (unsigned int t = 0; t < n_q_points; t++)
			  {
//				  cell_old_solu_Vectors[t] = cell_old_solu_vectors[t];
				  avg_old_tmp += cell_old_solu_Vectors[t];
				  if (cell_old_solu_Vectors[t] >= melt_point)
					  num_melt_point_old++;
//				  cell_solu_Vectors[t] = cell_solu_vectors[t];
//				  avg_tmp += cell_solu_vectors[t];
//				  if (cell_solu_vectors[t] >= melt_point)
//					  num_melt_point++;
//				  if(!(fabs(cell_old_solu_vectors[t] - 300) < 1e-3))
//				  {
//					  pcout<<"dof values"<<"("<<t<<")="<<cell_old_solu_Vectors_tmp[t] <<endl;
//					  pcout<<"fe  values"<<"("<<t<<")="<<cell_old_solu_vectors[t] <<endl;
//				  }

			  }
//			  avg_tmp /= n_q_points;

			  avg_old_tmp /= n_q_points;
			  avg_tmp = avg_old_tmp;
//			  num_melt_point /= n_q_points;
			  num_melt_point_old /= n_q_points;
			  num_melt_point = num_melt_point_old;

			  unsigned int old_material_id = old_cell_material[cnt_cells];
			  specific_heat_old = material_data.get_specific_heat (avg_old_tmp, old_material_id);
			  conductivity_old = material_data.get_conductivity (avg_old_tmp, old_material_id);
			  convectivity_old = material_data.get_convectivity (avg_old_tmp, old_material_id);
			  emissivity_old = material_data.get_emissivity (avg_old_tmp, old_material_id);

//
//			  if (old_material_id == 2)
//			  {// material id of last iteration (or old_material_id?) is powder
//				  {
//					  if (num_melt_point > 5/8)//(avg_tmp >= melt_point)
//						  cell->set_material_id(1);
//					  else
//						  if (cell->material_id() != 2)
//							  cell->set_material_id(2); //(0);
//				  }
//			  }
//			  else if (old_material_id == 1)
//			  {// material id of last iteration is liquid
//				  if (num_melt_point < 0.5)//(avg_tmp < melt_point)
//					  cell->set_material_id(0);
//				  else
//					  cell->set_material_id(1);
//			  }
//			  else if (old_material_id == 0)
//			  {// material id of last iteration is solid
//				  {
//					  if (num_melt_point > 4/8)//(avg_tmp >= melt_point)
//						  cell->set_material_id(1);
//					  else
//						  cell->set_material_id(0);
//				  }
//			  }

		    	Point<dim> centerPt = cell->center();
//		    	if (centerPt[2] <= limit)
		    	{// only consider the cell under the part height (activated cell)
		    		double local_x = (centerPt[0] - laserCenter[0])*cos(orientation) + (centerPt[1] - laserCenter[1])*sin(orientation),
		    					 local_y = -(centerPt[0] - laserCenter[0])*sin(orientation) + (centerPt[1] - laserCenter[1])*cos(orientation);
		    		if (local_x > 0) // point in local axis lie in back of the laser heat. In this circumstance, elongate the long radius of the ellipsoid
		    			a = 0.25*parameters.w;
		    		else
		    			a = parameters.w;
		    	    double distance = pow(local_x/(1*a), 2) + pow(local_y/(1*b), 2) +
		    	    							  pow((centerPt[2] - laserCenter[2])/(1*c), 2) - 1;
		    		if (distance <= 0)
		    		{// center point of current active cell fall into the defined surface
		    				cell->set_material_id(0); // set material id to solid()
		    		}
		    	}
		    	if (cell->material_id() == 1)
		    		cell->set_material_id(0);
		    	unsigned int material_id = cell->material_id();
		    	if(old_material_id == material_id)
		    	{
		    		specific_heat_Tnk = specific_heat_old;
		    		conductivity_Tnk = conductivity_old;
		    		convectivity_Tnk = convectivity_old;
		    		emissivity_Tnk = emissivity_old;
		    	}
		    	else
		    	{
		    		cell_material[cnt_cells] = material_id;
		    		initial_cell_material[cnt_cells] = material_id;
		    		specific_heat_Tnk = material_data.get_specific_heat (avg_tmp, material_id);
		    		conductivity_Tnk = material_data.get_conductivity (avg_tmp, material_id);
		    		convectivity_Tnk = material_data.get_convectivity (avg_tmp, material_id);
		    		emissivity_Tnk = material_data.get_emissivity (avg_tmp, material_id);
		    	}

			  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			  {
				  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
					  for (unsigned int j=0; j<dofs_per_cell; ++j)
					  {
						  cell_mass_matrix(i, j) += (fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point))*fe_values.JxW(q_point);
						  cell_laplace_matrix(i, j) += (fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point))*fe_values.JxW(q_point);
					  }
					  cell_rhs(i) += (rhs_values [q_point]*theta + rhs_old_values [q_point]*(1-theta))*
							   time_step*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point);	// cell_rhs = Tau_n*theta*Fn+Tau_n*(1-theta)*Fn-1
				  }
			  }

			  double coeff_cell_mass_matrix = density*specific_heat_Tnk,
					  	  coeff_cell_laplace_matrix = conductivity_Tnk*time_step*theta;
			  cell_matrix.add(coeff_cell_mass_matrix, cell_mass_matrix, coeff_cell_laplace_matrix, cell_laplace_matrix);
			  	  // cell_matrix = M(Tn) + Tau_n*theta*K(Tn)

//			  double coeff_cell_mass_matrix_old = density*specific_heat_old;
			  double coeff_cell_laplace_matrix_old = -time_step*(1-theta)*conductivity_old;
			  Vector<double> temp_mass(n_q_points);
			  Vector<double> temp_laplace(n_q_points);
			  cell_mass_matrix.vmult(temp_mass, cell_old_solu_Vectors);
			  cell_laplace_matrix.vmult(temp_laplace, cell_old_solu_Vectors);
	          cell_rhs.add(coeff_cell_mass_matrix, temp_mass, coeff_cell_laplace_matrix_old, temp_laplace);
	          	  // cell_rhs += M(Tn)*Tn-1 - Tau_n*(1-theta)*K(Tn-1)*Tn-1

//	          Point<dim> centerPt = cell->center();
	          if (centerPt[dim - 1] > thickness*(layer_id - 1))
	          {
	        	  // Apply Boundary conditions
	        	  //
	        	  //
	        	  face_cell_mass_matrix = 0;
	        	  face_cell_rhs = 0;

				  double coeff_face_cell_rhs_conv = time_step*Tamb*(theta*convectivity_Tnk + (1-theta)*convectivity_old);
				  double coeff_face_cell_rhs_rad =time_step*Tamb*Stephan_Boltzmann*(theta*emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
																						  +
																						  (1-theta)*emissivity_old*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
				  double coeff_face_cell_rhs = coeff_face_cell_rhs_conv + coeff_face_cell_rhs_rad;

				  double coeff_face_cell_matrix_conv = convectivity_Tnk*time_step*theta;
				  double coeff_face_cell_matrix_rad = emissivity_Tnk*Stephan_Boltzmann*(avg_tmp + Tamb) // avg_tmp should be replaced with Tnk
																						*(avg_tmp*avg_tmp + Tamb*Tamb)*time_step*theta;
				  double coeff_face_cell_matrix = coeff_face_cell_matrix_conv + coeff_face_cell_matrix_rad;

				  double coeff_face_cell_rhs_mass_conv = -time_step*(1-theta)*convectivity_old;
				  double coeff_face_cell_rhs_mass_rad = - time_step*(1-theta)*emissivity_old*Stephan_Boltzmann*(avg_old_tmp + Tamb)
																					  *(avg_old_tmp*avg_old_tmp + Tamb*Tamb);
				  double coeff_face_cell_rhs_mass = coeff_face_cell_rhs_mass_conv + coeff_face_cell_rhs_mass_rad;

				  for (unsigned int face_number=0; face_number <
											  GeometryInfo<dim>::faces_per_cell; ++face_number)
				  {
					  // Tests to select the cell faces which belong to the convection and radiation boundary
					  if (cell->face(face_number)->at_boundary()
									  && (cell->face(face_number)->boundary_id() == 1 ))
					  {// convection boundary & radiation boundary
						  // Term to be added in the RHS
						  hp_fe_face_values.reinit (cell, face_number);
						  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();

						  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
						  {
							  // Computation and storage of the value of the shape functions on boundary face integration points
							  for (unsigned int i=0; i<dofs_per_cell; ++i)
							  {
								  for (unsigned int j=0; j<dofs_per_cell; ++j)
								  {
									  face_cell_mass_matrix(i, j) += (fe_face_values.shape_value(i, q_point)*fe_face_values.shape_value(j, q_point))*fe_face_values.JxW(q_point);
								  }
								  face_cell_rhs(i) += fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
								  // face_cell_rhs = Tau_n*theta*Tamb*Vconv_n + Tau_n*(1-theta)*Tamb*Vconv_n-1 +
								  //	  						Tau_n*theta*Tamb*Vrad_n + Tau_n*(1-theta)*Tamb*Vrad_n-1
							  }
						  }

						  cell_matrix.add(coeff_face_cell_matrix, face_cell_mass_matrix);

						  Vector<double> temp_mass_convection(dofs_per_cell);
						  face_cell_mass_matrix.vmult(temp_mass_convection, cell_old_solu_Vectors);
						  cell_rhs.add(coeff_face_cell_rhs_mass, temp_mass_convection);
						  cell_rhs.add(coeff_face_cell_rhs, face_cell_rhs);
					  }

//					  // apply the input heat flux boundary condition: ID = 1
//					  if (cell->face(face_number)->at_boundary()
//									  && (cell->face(face_number)->boundary_id() == 1) && time <= layer_end_time)
//					  {// input heat flux boundary
//						  // Term to be added in the RHS
//						  hp_fe_face_values.reinit (cell, face_number);
//						  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();
//
//						  std::vector<double>  input_heat_flux_current_values (n_face_q_points),
//														   input_heat_flux_old_values (n_face_q_points);
//						  input_heat_flux_func.set_time(time);
//						  input_heat_flux_func.value_list (fe_face_values.get_quadrature_points(),
//																			input_heat_flux_current_values);
//
//
//						  input_heat_flux_func_old.set_time(time - time_step);
//						  input_heat_flux_func_old.value_list (fe_face_values.get_quadrature_points(),
//																			input_heat_flux_old_values);
//
//						  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
//						  {
//							  // input heat flux value.
//							  // Computation and storage of the value of the shape functions on boundary face integration points
//							  double coeff_heat_flux_at_q_point = time_step*theta*input_heat_flux_current_values[q_point] +
//																								  time_step*(1 - theta)*input_heat_flux_old_values[q_point];
//							  for (unsigned int i=0; i<dofs_per_cell; ++i)
//							  {
//								  cell_rhs(i) += coeff_heat_flux_at_q_point*fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
//							  }
//						  }
//					  }

				  }

	          }
			  cell->get_dof_indices (local_dof_indices);
//			  for (unsigned int i=0; i<dofs_per_cell; ++i)
//			  {
//				  for (unsigned int j=0; j<dofs_per_cell; ++j)
//					  system_matrix.add (local_dof_indices[i],
//							  	  	  	  	  	  	  local_dof_indices[j],
//													  cell_matrix(i,j));
//				  system_rhs(local_dof_indices[i]) += cell_rhs(i);
//			  }
			  constraints.distribute_local_to_global (cell_matrix, cell_rhs,
	  		                                              local_dof_indices,
	  		                                              system_matrix, system_rhs); // this doesn't work for hp elements, don't know why


		  }
	  	  }
	  }

	  system_matrix.compress(VectorOperation::add);
	  system_rhs.compress(VectorOperation::add);
////	  std::cout <<"number of active cells in thermal analysis: "<<active_cnt_cells<<std::endl;
  }


  template <int dim>
  void HeatEquation<dim>::update_assemble_system_test(const Vector<double> Temp_n_k, const Vector<double> initial_cell_material)
  {	//"initial_cell_material" is used to balance the initial assembled Temp_n_k, which is T_old
	  TimerOutput::Scope timer_section(computing_timer, "Update Assembling Test");

//	  hp::FEValues<dim> hp_fe_values (fe_collection, quadrature_collection,
//			  	  	  	  	  	  update_values | update_gradients |
//								  update_quadrature_points | update_JxW_values);
	  // Finite element evaluated in quadrature points of the faces of a cell.
	  hp::FEFaceValues<dim> hp_fe_face_values(fe_collection, face_quadrature_collection,
													update_values | update_quadrature_points |
													update_quadrature_points | update_JxW_values);

	  LA::MPI::SparseMatrix update_system_matrix;
//	  update_system_matrix.reinit (locally_owned_dofs,
//	                          locally_owned_dofs,
//	                          sparsity_pattern,
//	                          mpi_communicator);
	  update_system_matrix.reinit(system_matrix);
	  LA::MPI::Vector update_system_rhs(locally_owned_dofs,mpi_communicator);
	  update_system_matrix = 0;
	  update_system_rhs = 0;

	  const unsigned int           n_q_points    = quadrature_collection[0].size();		// n_q_points = 8
	  const unsigned int 			n_face_q_points = face_quadrature_collection[0].size(); // quadrature points on faces = 4
	  const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell; // dofs_per_active_cell = 8

	  FullMatrix<double> 			cell_matrix(dofs_per_active_cell, dofs_per_active_cell);
//			  	  	  	  	  	  	  	  	  	    cell_mass_matrix(dofs_per_active_cell, dofs_per_active_cell),
//													cell_laplace_matrix(dofs_per_active_cell, dofs_per_active_cell);
	  FullMatrix<double>			    face_cell_mass_matrix (dofs_per_active_cell, dofs_per_active_cell);
	  Vector<double>	 		    	cell_rhs(dofs_per_active_cell);
	  Vector<double>      	        	face_cell_rhs (dofs_per_active_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_active_cell);

//	  std::vector<double> 		cell_solu_vectors(n_q_points), cell_old_solu_vectors(n_q_points);
	  Vector<double> 			cell_solu_Vectors(n_q_points), cell_old_solu_Vectors(n_q_points);
	  std::vector<double> 		cell_diff_solu_vectors(n_q_points);

	  double     specific_heat_Tnk, specific_heat_initial,
			  	  	  conductivity_Tnk, conductivity_initial,
					  convectivity_Tnk, convectivity_initial,
					  emissivity_Tnk, emissivity_initial;

	  MaterialData<dim> material_data (parameters);
	  double melt_point = parameters.melt_point;// = 1650 + 273;
	  double density = parameters.density;
	  const double Stephan_Boltzmann = 5.67e-8;
	  unsigned int cnt_cells (0);
	  unsigned int cnt_skiped_cells (0);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  endc = dof_handler.end();
	  for (; cell!=endc; ++cell, ++cnt_cells)
	  {
		  if (cell->is_locally_owned())
		  {
			  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
			  if (dofs_per_cell != 0)
			  {
	//			  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: assemble active cell");

//				  hp_fe_values.reinit(cell);
//				  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

				  cell->get_dof_values(old_solution, cell_old_solu_Vectors);
				  cell->get_dof_values(Temp_n_k, cell_solu_Vectors);

				  for (unsigned int i=0; i<n_q_points; ++i)
				  {
					  cell_diff_solu_vectors[i] = cell_solu_Vectors[i] - cell_old_solu_Vectors[i];
				  }

				  double sum = std::accumulate(cell_diff_solu_vectors.begin(), cell_diff_solu_vectors.end(), 0.0);
				  double mean = sum / cell_diff_solu_vectors.size();

				  if (fabs(mean) < 0.1)	// if the average temperature of current cell is almost the same with temperature of old cell, skip assemble this cell and continue next cell
				  {
	//				  std::cout <<"mean = "<<mean<< std::endl;
					  cnt_skiped_cells++;
					  continue;
				  }

				  cell_matrix = 0;
//				  cell_mass_matrix = 0;
//				  cell_laplace_matrix = 0;
				  cell_rhs = 0;

				  double avg_tmp = 0., avg_old_tmp = 0.;
				  double num_melt_point = 0;//, num_melt_point_old = 0;

				  for (unsigned int t = 0; t < n_q_points; t++)
				  {
//					  cell_old_solu_Vectors[t] = cell_old_solu_vectors[t];
					  avg_old_tmp += cell_old_solu_Vectors[t];
					  avg_tmp += cell_solu_Vectors[t];
					  if (cell_solu_Vectors[t] >= melt_point)
						  num_melt_point++;
				  }
				  avg_tmp /= n_q_points;

				  avg_old_tmp /= n_q_points;
				  num_melt_point /= n_q_points;
	//			  num_melt_point_old /= n_q_points;

				  specific_heat_initial = material_data.get_specific_heat (avg_old_tmp, initial_cell_material[cnt_cells]);
				  conductivity_initial = material_data.get_conductivity (avg_old_tmp, initial_cell_material[cnt_cells]);
				  convectivity_initial = material_data.get_convectivity (avg_old_tmp, initial_cell_material[cnt_cells]);
				  emissivity_initial = material_data.get_emissivity (avg_old_tmp, initial_cell_material[cnt_cells]);

				  if (num_melt_point > 5/8)//(avg_tmp >= melt_point)
					  cell->set_material_id(1);
//				  else
//				  {
//					  cell->set_material_id(0); //
//				  }
				  unsigned int material_id = cell->material_id();
				  cell_material[cnt_cells] = material_id;
				  specific_heat_Tnk = material_data.get_specific_heat (avg_tmp, material_id);
				  conductivity_Tnk = material_data.get_conductivity (avg_tmp, material_id);
				  convectivity_Tnk = material_data.get_convectivity (avg_tmp, material_id);
				  emissivity_Tnk = material_data.get_emissivity (avg_tmp, material_id);

//				  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
//				  {
//	//				  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: mass&laplace matrix");
//					  for (unsigned int i=0; i<dofs_per_cell; ++i)
//					  {
//						  for (unsigned int j=0; j<dofs_per_cell; ++j)
//						  {
//							  cell_mass_matrix(i, j) += (fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point))*fe_values.JxW(q_point);
//							  cell_laplace_matrix(i, j) += (fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point))*fe_values.JxW(q_point);
//						  }
//	//					  cell_rhs(i) += (rhs_values [q_point]*theta //+ rhs_old_values [q_point]*(1-theta)
//	//					  	  	  	  	  	  	  )*time_step*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point);	// cell_rhs = Tau_n*theta*Fn+Tau_n*(1-theta)*Fn-1
//					  }
//				  }

//				  cell_mass_matrix = cell_mass_matrix_list[cnt_cells];
//				  cell_laplace_matrix = cell_laplace_matrix_list[cnt_cells];

				  double coeff_cell_mass_matrix = density*(specific_heat_Tnk - specific_heat_initial),
							  coeff_cell_laplace_matrix = (conductivity_Tnk - conductivity_initial)*time_step*theta;
				  cell_matrix.add(coeff_cell_mass_matrix, cell_mass_matrix_list[cnt_cells], coeff_cell_laplace_matrix,cell_laplace_matrix_list[cnt_cells]);
				  // cell_matrix = M(Tn) + Tau_n*theta*K(Tn)

				  Vector<double> temp_mass(n_q_points);
				  cell_mass_matrix_list[cnt_cells].vmult(temp_mass, cell_old_solu_Vectors);
				  cell_rhs.add(coeff_cell_mass_matrix, temp_mass);

//				  Point<dim> centerPt = cell->center();
				  if (layer_id > 5)		//(centerPt[dim - 1] > thickness*(layer_id - 1))
				  {
					  // Apply Boundary conditions
					  //
					  //
					  face_cell_mass_matrix = 0;
					  face_cell_rhs = 0;

					  double coeff_face_cell_rhs_conv = time_step*Tamb*theta*(convectivity_Tnk - convectivity_initial);
					  double coeff_face_cell_rhs_rad = time_step*Tamb*Stephan_Boltzmann*theta
																	  *(emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
																		-
																		emissivity_initial*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
					  double coeff_face_cell_rhs = coeff_face_cell_rhs_conv + coeff_face_cell_rhs_rad;

					  double coeff_face_cell_matrix_conv = coeff_face_cell_rhs_conv/Tamb; //time_step*theta*(convectivity_Tnk - convectivity_initial);
					  double coeff_face_cell_matrix_rad = coeff_face_cell_rhs_rad/Tamb;
		//					  	  	  	  	  	  	  	  	  	  	  	  time_step*Stephan_Boltzmann*theta
		//					  	  	  	  	  	  	  	  	  	  	  	  *(emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
		//								  	  	  	  	  	  	  	  	  	 -
		//																	 emissivity_initial*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
					  double coeff_face_cell_matrix = coeff_face_cell_matrix_conv + coeff_face_cell_matrix_rad;

					  for (unsigned int face_number=0; face_number <
												  GeometryInfo<dim>::faces_per_cell; ++face_number)
					  {
						  // Tests to select the cell faces which belong to the convection and radiation boundary: ID=0
						  if (cell->face(face_number)->at_boundary()
										  && (cell->face(face_number)->boundary_id() == 1 ))
						  {//update convection boundary & radiation boundary
							  // Term to be added in the RHS
							  hp_fe_face_values.reinit (cell, face_number);
							  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();

							  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
							  {
								  // Computation and storage of the value of the shape functions on boundary face integration points
								  for (unsigned int i=0; i<dofs_per_cell; ++i)
								  {
									  for (unsigned int j=0; j<dofs_per_cell; ++j)
									  {
										  face_cell_mass_matrix(i, j) += (fe_face_values.shape_value(i, q_point)*fe_face_values.shape_value(j, q_point))*fe_face_values.JxW(q_point);
									  }
									  face_cell_rhs(i) += fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
									  // face_cell_rhs = Tau_n*theta*Tamb*Vconv_n + Tau_n*(1-theta)*Tamb*Vconv_n-1 +
									  //	  						Tau_n*theta*Tamb*Vrad_n + Tau_n*(1-theta)*Tamb*Vrad_n-1
								  }
							  }

							  cell_matrix.add(coeff_face_cell_matrix, face_cell_mass_matrix);
							  cell_rhs.add(coeff_face_cell_rhs, face_cell_rhs);
						  }

					  }
				  }

				  cell->get_dof_indices (local_dof_indices);
//				  for (unsigned int i=0; i<dofs_per_cell; ++i)
//				  {
//	//	    		  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: add cell matrix");
//					  for (unsigned int j=0; j<dofs_per_cell; ++j)
//						  update_system_matrix.add (local_dof_indices[i],
//																	  local_dof_indices[j],
//																	  cell_matrix(i,j));
//					  update_system_rhs(local_dof_indices[i]) += cell_rhs(i);
//				  }
				  constraints.distribute_local_to_global (cell_matrix, cell_rhs,
		  		                                              local_dof_indices,
															  update_system_matrix, update_system_rhs); // this doesn't work for hp elements, don't know why


			  }

		  }
	  }

//	  std::cout <<"=================== "<< std::endl;
//	  std::cout <<"cnt_skiped_cells = "<<cnt_skiped_cells<< std::endl;

	  update_system_matrix.compress(VectorOperation::add);
	  update_system_rhs.compress(VectorOperation::add);

	  system_matrix.add(1.0, update_system_matrix);
	  system_rhs.add(1.0, update_system_rhs);

	  system_matrix.compress(VectorOperation::add);
	  system_rhs.compress(VectorOperation::add);

	  update_system_matrix.clear();
	  update_system_rhs.clear();
//      constraints.condense(system_matrix, system_rhs);
  }

  template <int dim>
  void HeatEquation<dim>::update_assemble_system(const Vector<double> Temp_n_k, const Vector<double> initial_cell_material)
  {	//"initial_cell_material" is used to balance the initial assembled Temp_n_k, which is T_old
	  TimerOutput::Scope timer_section(computing_timer, "Update Assembling");

	  hp::FEValues<dim> hp_fe_values (fe_collection, quadrature_collection,
			  	  	  	  	  	  update_values | update_gradients |
								  update_quadrature_points | update_JxW_values);
	  // Finite element evaluated in quadrature points of the faces of a cell.
	  hp::FEFaceValues<dim> hp_fe_face_values(fe_collection, face_quadrature_collection,
													update_values | update_quadrature_points |
													update_quadrature_points | update_JxW_values);

	  LA::MPI::SparseMatrix update_system_matrix;
//	  update_system_matrix.reinit (locally_owned_dofs,
//	                          locally_owned_dofs,
//	                          sparsity_pattern,
//	                          mpi_communicator);
	  update_system_matrix.reinit(system_matrix);
	  LA::MPI::Vector update_system_rhs(locally_owned_dofs,mpi_communicator);
	  update_system_matrix = 0;
	  update_system_rhs = 0;

	  const unsigned int           n_q_points    = quadrature_collection[0].size();		// n_q_points = 8
	  const unsigned int 			n_face_q_points = face_quadrature_collection[0].size(); // quadrature points on faces = 4
	  const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell; // dofs_per_active_cell = 8

	  FullMatrix<double> 			cell_matrix(dofs_per_active_cell, dofs_per_active_cell),
			  	  	  	  	  	  	  	  	  	    cell_mass_matrix(dofs_per_active_cell, dofs_per_active_cell),
													cell_laplace_matrix(dofs_per_active_cell, dofs_per_active_cell);
	  FullMatrix<double>			    face_cell_mass_matrix (dofs_per_active_cell, dofs_per_active_cell);
	  Vector<double>	 		    	cell_rhs(dofs_per_active_cell);
	  Vector<double>      	        	face_cell_rhs (dofs_per_active_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_active_cell);

//	  EquationData::InputHeatFlux<dim> rhs_function(parameters,
//			  source_type, scan_velocity, segment_start_time, segment_length, orientation, segment_start_point, segment_end_point);
//	  EquationData::RightHandSide<dim> rhs_function;
//	  std::vector<double>  rhs_values (n_q_points), rhs_old_values (n_q_points);

//	  std::vector<double> 		cell_solu_vectors(n_q_points), cell_old_solu_vectors(n_q_points);
	  Vector<double> 			cell_solu_Vectors(n_q_points), cell_old_solu_Vectors(n_q_points);
	  std::vector<double> 		cell_diff_solu_vectors(n_q_points);

	  double     specific_heat_Tnk, specific_heat_initial,
			  	  	  conductivity_Tnk, conductivity_initial,
					  convectivity_Tnk, convectivity_initial,
					  emissivity_Tnk, emissivity_initial;

	  MaterialData<dim> material_data (parameters);
	  double melt_point = parameters.melt_point;// = 1650 + 273;
	  double density = parameters.density;
	  const double Stephan_Boltzmann = 5.67e-8;
	  unsigned int cnt_cells (0);
	  unsigned int cnt_skiped_cells (0);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  endc = dof_handler.end();
	  for (; cell!=endc; ++cell, ++cnt_cells)
	  {
		  if (cell->is_locally_owned())
		  {
			  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
			  if (dofs_per_cell != 0)
			  {
	//			  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: assemble active cell");

				  hp_fe_values.reinit(cell);
				  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

	//			  rhs_function.set_time(time);
	//			  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);	// Qn ---> rhs_values
	//			  rhs_function.set_time(time - time_step);
	//			  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_old_values);		// Qn-1 ---> rhs_old_values

				  cell->get_dof_values(old_solution, cell_old_solu_Vectors);
				  cell->get_dof_values(Temp_n_k, cell_solu_Vectors);
//				  fe_values.get_function_values(old_solution, cell_old_solu_vectors);		// Tn-1  ---> cell_old_solu_vectors
//				  fe_values.get_function_values(Temp_n_k, cell_solu_vectors);	// Tnk  ---> cell_solu_vectors

				  for (unsigned int i=0; i<n_q_points; ++i)
				  {
					  cell_diff_solu_vectors[i] = cell_solu_Vectors[i] - cell_old_solu_Vectors[i];
				  }

				  double sum = std::accumulate(cell_diff_solu_vectors.begin(), cell_diff_solu_vectors.end(), 0.0);
				  double mean = sum / cell_diff_solu_vectors.size();
	//			  std::vector<double> diff(cell_old_solu_vectors.size());
	//			  std::transform(cell_old_solu_vectors.begin(), cell_old_solu_vectors.end(), diff.begin(),
	//			  					                 std::bind2nd(std::minus<double>(), mean));
	//			  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	//			  double stdev = std::sqrt(sq_sum / cell_old_solu_vectors.size());
	//			  std::cout <<"mean = "<<mean<< std::endl;
	//			  std::cout <<"stdev = "<<stdev<< std::endl;
				  if (fabs(mean) < 0.1)	// if the average temperature of current cell is almost the same with temperature of old cell, skip assemble this cell and continue next cell
				  {
	//				  std::cout <<"mean = "<<mean<< std::endl;
					  cnt_skiped_cells++;
					  continue;
				  }

				  cell_matrix = 0;
				  cell_mass_matrix = 0;
				  cell_laplace_matrix = 0;
				  cell_rhs = 0;

				  double avg_tmp = 0., avg_old_tmp = 0.;
				  double num_melt_point = 0;//, num_melt_point_old = 0;

				  for (unsigned int t = 0; t < n_q_points; t++)
				  {
//					  cell_old_solu_Vectors[t] = cell_old_solu_vectors[t];
					  avg_old_tmp += cell_old_solu_Vectors[t];
	//				  if (cell_old_solu_vectors[t] >= melt_point)
	//					  num_melt_point_old++;
//					  cell_solu_Vectors[t] = cell_solu_vectors[t];
					  avg_tmp += cell_solu_Vectors[t];
					  if (cell_solu_Vectors[t] >= melt_point)
						  num_melt_point++;
				  }
				  avg_tmp /= n_q_points;

				  avg_old_tmp /= n_q_points;
				  num_melt_point /= n_q_points;
	//			  num_melt_point_old /= n_q_points;

				  specific_heat_initial = material_data.get_specific_heat (avg_old_tmp, initial_cell_material[cnt_cells]);
				  conductivity_initial = material_data.get_conductivity (avg_old_tmp, initial_cell_material[cnt_cells]);
				  convectivity_initial = material_data.get_convectivity (avg_old_tmp, initial_cell_material[cnt_cells]);
				  emissivity_initial = material_data.get_emissivity (avg_old_tmp, initial_cell_material[cnt_cells]);

	//			  Point<dim> pt_cell_center = cell->center();

	//			  unsigned int old_material_id = old_cell_material[cnt_cells];
	//			  if (old_material_id == 2)
	//			  {// material id of last iteration (or old_material_id?) is powder
	//				  {
	//					  if (num_melt_point > 5/8)//(avg_tmp >= melt_point)
	//						  cell->set_material_id(1);
	//					  else
	//						  if (cell->material_id() != 2)
	//							  cell->set_material_id(1); //(0);
	//				  }
	//			  }
	//			  else if (old_material_id == 1)
	//			  {// material id of last iteration is liquid
	//				  if (num_melt_point < 0.5)//(avg_tmp < melt_point)
	//					  cell->set_material_id(0);
	//				  else
	//					  cell->set_material_id(1);
	//			  }
	//			  else if (old_material_id == 0)
	//			  {// material id of last iteration is solid
	//				  {
	//					  if (num_melt_point > 4/8)//(avg_tmp >= melt_point)
	//						  cell->set_material_id(1);
	//					  else
	//						  cell->set_material_id(0);
	//				  }
	//			  }

				  if (num_melt_point > 5/8)//(avg_tmp >= melt_point)
					  cell->set_material_id(1);
	//			  else
	//			  {
	//				  cell->set_material_id(0); //
	//			  }
				  unsigned int material_id = cell->material_id();
				  cell_material[cnt_cells] = material_id;
				  specific_heat_Tnk = material_data.get_specific_heat (avg_tmp, material_id);
				  conductivity_Tnk = material_data.get_conductivity (avg_tmp, material_id);
				  convectivity_Tnk = material_data.get_convectivity (avg_tmp, material_id);
				  emissivity_Tnk = material_data.get_emissivity (avg_tmp, material_id);

				  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
				  {
	//				  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: mass&laplace matrix");
					  for (unsigned int i=0; i<dofs_per_cell; ++i)
					  {
						  for (unsigned int j=0; j<dofs_per_cell; ++j)
						  {
							  cell_mass_matrix(i, j) += (fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point))*fe_values.JxW(q_point);
							  cell_laplace_matrix(i, j) += (fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point))*fe_values.JxW(q_point);
						  }
	//					  cell_rhs(i) += (rhs_values [q_point]*theta //+ rhs_old_values [q_point]*(1-theta)
	//					  	  	  	  	  	  	  )*time_step*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point);	// cell_rhs = Tau_n*theta*Fn+Tau_n*(1-theta)*Fn-1
					  }
				  }

				  double coeff_cell_mass_matrix = density*(specific_heat_Tnk - specific_heat_initial),
							  coeff_cell_laplace_matrix = (conductivity_Tnk - conductivity_initial)*time_step*theta;
				  cell_matrix.add(coeff_cell_mass_matrix, cell_mass_matrix, coeff_cell_laplace_matrix,cell_laplace_matrix);
				  // cell_matrix = M(Tn) + Tau_n*theta*K(Tn)

				  Vector<double> temp_mass(n_q_points);
				  cell_mass_matrix.vmult(temp_mass, cell_old_solu_Vectors);
				  cell_rhs.add(coeff_cell_mass_matrix, temp_mass);

				  Point<dim> centerPt = cell->center();
				  if (centerPt[dim - 1] > thickness*(layer_id - 1))
				  {
					  // Apply Boundary conditions
					  //
					  //
					  face_cell_mass_matrix = 0;
					  face_cell_rhs = 0;

					  double coeff_face_cell_rhs_conv = time_step*Tamb*theta*(convectivity_Tnk - convectivity_initial);
					  double coeff_face_cell_rhs_rad = time_step*Tamb*Stephan_Boltzmann*theta
																	  *(emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
																		-
																		emissivity_initial*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
					  double coeff_face_cell_rhs = coeff_face_cell_rhs_conv + coeff_face_cell_rhs_rad;

					  double coeff_face_cell_matrix_conv = coeff_face_cell_rhs_conv/Tamb; //time_step*theta*(convectivity_Tnk - convectivity_initial);
					  double coeff_face_cell_matrix_rad = coeff_face_cell_rhs_rad/Tamb;
		//					  	  	  	  	  	  	  	  	  	  	  	  time_step*Stephan_Boltzmann*theta
		//					  	  	  	  	  	  	  	  	  	  	  	  *(emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
		//								  	  	  	  	  	  	  	  	  	 -
		//																	 emissivity_initial*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
					  double coeff_face_cell_matrix = coeff_face_cell_matrix_conv + coeff_face_cell_matrix_rad;

					  for (unsigned int face_number=0; face_number <
												  GeometryInfo<dim>::faces_per_cell; ++face_number)
					  {
						  // Tests to select the cell faces which belong to the convection and radiation boundary: ID=0
						  if (cell->face(face_number)->at_boundary()
										  && (cell->face(face_number)->boundary_id() == 1 ))
						  {//update convection boundary & radiation boundary
							  // Term to be added in the RHS
							  hp_fe_face_values.reinit (cell, face_number);
							  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();

							  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
							  {
								  // Computation and storage of the value of the shape functions on boundary face integration points
								  for (unsigned int i=0; i<dofs_per_cell; ++i)
								  {
									  for (unsigned int j=0; j<dofs_per_cell; ++j)
									  {
										  face_cell_mass_matrix(i, j) += (fe_face_values.shape_value(i, q_point)*fe_face_values.shape_value(j, q_point))*fe_face_values.JxW(q_point);
									  }
									  face_cell_rhs(i) += fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
									  // face_cell_rhs = Tau_n*theta*Tamb*Vconv_n + Tau_n*(1-theta)*Tamb*Vconv_n-1 +
									  //	  						Tau_n*theta*Tamb*Vrad_n + Tau_n*(1-theta)*Tamb*Vrad_n-1
								  }
							  }

							  cell_matrix.add(coeff_face_cell_matrix, face_cell_mass_matrix);
							  cell_rhs.add(coeff_face_cell_rhs, face_cell_rhs);
						  }

					  }
				  }

				  cell->get_dof_indices (local_dof_indices);
//				  for (unsigned int i=0; i<dofs_per_cell; ++i)
//				  {
//	//	    		  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: add cell matrix");
//					  for (unsigned int j=0; j<dofs_per_cell; ++j)
//						  update_system_matrix.add (local_dof_indices[i],
//																	  local_dof_indices[j],
//																	  cell_matrix(i,j));
//					  update_system_rhs(local_dof_indices[i]) += cell_rhs(i);
//				  }
				  constraints.distribute_local_to_global (cell_matrix, cell_rhs,
		  		                                              local_dof_indices,
															  update_system_matrix, update_system_rhs); // this doesn't work for hp elements, don't know why


			  }

		  }
	  }

//	  std::cout <<"=================== "<< std::endl;
//	  std::cout <<"cnt_skiped_cells = "<<cnt_skiped_cells<< std::endl;

	  update_system_matrix.compress(VectorOperation::add);
	  update_system_rhs.compress(VectorOperation::add);

	  system_matrix.add(1.0, update_system_matrix);
	  system_rhs.add(1.0, update_system_rhs);

	  system_matrix.compress(VectorOperation::add);
	  system_rhs.compress(VectorOperation::add);

	  update_system_matrix.clear();
	  update_system_rhs.clear();
//      constraints.condense(system_matrix, system_rhs);

//      // Dirichlet boundary: ID = 2
//      { // Do not apply Dirichlet boudary when printing the first layer, otherwise the bottom cell will not become solid cell
//    	  TimerOutput::Scope timer_section(computing_timer, "Update Assembling: apply boundary");
//    	  EquationData::BoundaryValues<dim> boundary_values_function;
////    	  boundary_values_function.set_time(time);
//
//    	  std::map<types::global_dof_index, double> boundary_values;
//    	  VectorTools::interpolate_boundary_values(dof_handler,
//                                                       2,
//                                                       boundary_values_function,
//                                                       boundary_values);
//
//    	  MatrixTools::apply_boundary_values(boundary_values,
//                                                 system_matrix,
//                                                 solution,
//                                                 system_rhs);
//      }
  }

  template<int dim>
  void HeatEquation<dim>::solve_time_step()
  {
	  TimerOutput::Scope timer_section(computing_timer, "Solve");

	  LA::MPI::Vector distributed_solution (locally_owned_dofs, mpi_communicator);
	  distributed_solution = solution;

	  SolverControl           solver_control (dof_handler.n_dofs(), 1e-16);
//	  	                                            1e-16*system_rhs.l2_norm());

#ifdef USE_PETSC_LA
	    	    PETScWrappers::SolverCG cg (solver_control,
	    	                                mpi_communicator);
	    	    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
//	    	    cout<<"solving use PETSC"<<std::endl;
#else
//	    	    TrilinosWrappers::SolverCG cg(solver_control);
	    	    SolverCG<TrilinosWrappers::MPI::Vector>  cg (solver_control);
	    	    TrilinosWrappers::PreconditionAMG preconditioner;
	    	    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
	    	    preconditioner.initialize(system_matrix, additional_data);
#endif
	  cg.solve(system_matrix, distributed_solution, system_rhs,
             preconditioner);
//	  solution = distributed_solution;
//	  constraints.distribute (solution);
	  constraints.distribute (distributed_solution);
	  solution = distributed_solution;


	  for (unsigned int index = 0; index < solution.size(); index++)
	  {
		  if (solution[index] < Tinit)
			  solution[index] = Tinit;
	  }

//	  pcout << "     " << solver_control.last_step()
//	                  << " CG iterations." << std::endl;

  }

  double compute_relaxation_parameter(const double pre_error, const double current_error)
  {
	  double gamma = 1;
	  if (current_error < pre_error)
	  {
		  if (current_error > 0.5*pre_error)
			  gamma = 0.951;
		  else
		  {
			  if (fabs(current_error/pre_error) < 1e-3)
				  gamma = 0.99;//0.91;
			  else
				  gamma = (1- current_error/pre_error);
		  }
	  }
	  else
	  {// current_error >= pre_error
		  if (fabs(pre_error/current_error) < 1e-3)
			  gamma = 0.98;//0.92;
		  else
			  gamma = 1- pre_error/current_error;//0.91;
	  }
//	  gamma = 0.5;
	  if (fabs(gamma - 1.0) < 1e-3)
		  gamma = 0.95;
	  return gamma;
  }

  int* random(int n)
  {  //n为随机数大小, 随机产生[0, n)之间的数，半开半闭区间
	  std::vector<int>a;  //储存所有可能的随机数
	  int* randnum = new int[n];
	  //结果序列
	  for (int i = 0; i < n; i++)
	  {
		  if (i == 0)
			  a.push_back(n/2.0);
		  else
		  a.push_back(i);
	  }
	  //生成所有可能随机数
	  for (int i = 0; i < n; i++)
	  {
		  int choice = rand() % a.size(); //随机产生下标
		  randnum[i] = a[choice];
		  swap(a[choice], a[a.size()-1]);	  //交换已经生成的随机数和数组最后一个数
		  a.pop_back();  //删除已经生成的随机数
	  }
	  return randnum;
  }

  template<int dim>
  void HeatEquation<dim>::solve_relaxed_Picard()
  {
	  pcout << std::endl << "***Time step " << timestep_number << " at t=" << time
			  << std::endl;
	  pcout << "===========================================" << std::endl
                    << "Number of active cells: " << triangulation.n_active_cells()  << std::endl
					<< "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;

	  unsigned int iter_num = 0, max_iter_num = 1000;
	  double gamma = 0.5, epsilon = 0.01;//1e-7;
	  double pre_error = 10e9, current_error = 10e9;
	  Vector<double> Temp_n_k = old_solution;		//Tn,k = Tn-1  initial guess
	  Vector<double> Temp_n_k_plus_1(solution.size());//, tmp_solu = solution;

	  Vector<double> initial_cell_material = old_cell_material;
	  assemble_system_test(/*Temp_n_k, */initial_cell_material);
	  mesh_changed_flg = false;
//	  assemble_system(/*Temp_n_k, */initial_cell_material);
//	  assemble_system_test(/*Temp_n_k, */initial_cell_material);

	  LA::MPI::SparseMatrix 		system_matrix_tmp;
//	  system_matrix_tmp.reinit (locally_owned_dofs,
//	                          locally_owned_dofs,
//	                          sparsity_pattern,
//	                          mpi_communicator);
	  system_matrix_tmp.reinit(system_matrix);
	  system_matrix_tmp.copy_from(system_matrix);
	  LA::MPI::Vector system_rhs_tmp = system_rhs;

	  max_iter_num = 5;
	  while (iter_num < max_iter_num)
	  {
		  system_matrix.copy_from(system_matrix_tmp);
		  system_rhs = system_rhs_tmp;

//		  update_assemble_system(Temp_n_k, initial_cell_material);
		  update_assemble_system_test(Temp_n_k, initial_cell_material);
//		  tmp_solu = solution;
		  solve_time_step();		// solution -> Tn*

		  Temp_n_k_plus_1 = Temp_n_k;
		  Temp_n_k_plus_1.sadd(gamma, (1 - gamma), solution);	//T_n,k+1=r*Tn,k+(1-r)*Tn*
		  Vector<double> Temp_n_k_1 = Temp_n_k_plus_1, min_Temp_n_k_plus_1 = Temp_n_k_plus_1;
		  Temp_n_k_1.add(-1.0, Temp_n_k);

		  current_error = Temp_n_k_1.l2_norm();

		  double tmp_gamma = gamma;
		  double minimum_current_error = current_error;
//		  unsigned int minimum_choice = 0;
		  unsigned int choice = 0;
		  int* rand_num = random(100);
		  bool enter_flg = false;
		  while(timestep_number > 10 && current_error > 1*pre_error && pre_error < 10 && choice <100)
		  {
			  enter_flg = true;
			  gamma = rand_num[choice]/100.0;//1- pre_error/tmp_current_error;//compute_relaxation_parameter(pre_error, current_error);
			  Temp_n_k_plus_1 = Temp_n_k;
			  Temp_n_k_plus_1.sadd(gamma, (1 - gamma), solution);	//T_n,k+1=r*Tn,k+(1-r)*Tn*
			  Temp_n_k_1 = Temp_n_k_plus_1;
			  Temp_n_k_1.add(-1.0, Temp_n_k);

			  current_error = Temp_n_k_1.l2_norm();
			  if (current_error <= minimum_current_error)
			  {
				  minimum_current_error = current_error;
//				  minimum_choice = choice;
				  min_Temp_n_k_plus_1 = Temp_n_k_plus_1;
			  }

////			  tmp_gamma = gamma;
//			  std::cout << "    --- ++++++ minimum_current_error = "<<minimum_current_error <<" , current_error =  "<<current_error<<"				:gamma = " << gamma<< std::endl; // gamma cannot be 1, otherwise T_n,k = T_n,k+1, the next iteration will always converge!

			  if (current_error <= pre_error)
				  break;
			  choice++;
		  }
		  gamma = tmp_gamma;
		  current_error = minimum_current_error;
		  Temp_n_k_plus_1 = min_Temp_n_k_plus_1;

		  if (timestep_number > 10 && current_error > pre_error && pre_error < 10 || (enter_flg && current_error < 0.1 && fabs(current_error - pre_error) < 0.1))
		  {
			  pcout << "    ---INFORMAL 1 error = " <<  current_error <<"				:gamma = " << gamma<< std::endl; // gamma cannot be 1, otherwise T_n,k = T_n,k+1, the next iteration will always converge!
			  break;
		  }

//		  if (timestep_number == 13 || timestep_number == 22 || timestep_number == 48 || timestep_number == 46 || timestep_number == 106 || timestep_number == 130 || timestep_number == 137
//				  || timestep_number > 309)
//			  std::cout << "    --- l1_norm() = " <<  Temp_n_k_1.l1_norm() <<", current error = " <<  current_error <<"				:gamma = " << gamma<< std::endl; // gamma cannot be 1, otherwise T_n,k = T_n,k+1, the next iteration will always converge!


		  if (current_error < epsilon)// && compute_error_residual(Temp_n_k_plus_1) < 2e-3)
		  {
			  pcout << "    --- error = " <<  current_error <<"				:gamma = " << gamma<< std::endl; // gamma cannot be 1, otherwise T_n,k = T_n,k+1, the next iteration will always converge!
			  break;
		  }

		  if (fabs(pre_error - current_error) < 1e-2 || current_error < 1e-1 || iter_num > 35 || current_error/pre_error > 20)
		  {
			  if (current_error < 1e-1)
				  gamma = 0.951;
			  else if (current_error/pre_error > 20)
			  {
				  gamma = 0.95;
			  }
			  else
				  gamma = compute_relaxation_parameter(pre_error, current_error);

			  if (iter_num >35)
				  epsilon = 0.1;
			  if (iter_num > 100 && current_error < 10e-1)
			  {
				  pcout << "    ---INFORMAL 2 error = " <<  current_error <<"				:gamma = " << gamma<< std::endl;
				  break;
			  }
		  }

		  Temp_n_k = Temp_n_k_plus_1;
		  pre_error = current_error;
		  iter_num++;
	  }
	  pcout << "    +++ Converged in " << iter_num << " iterations." << std::endl;
	  solution = Temp_n_k_plus_1;	//Tn=Tn,k+1

	  if (iter_num == max_iter_num)
	  {
//		  pcout << "    !!!! Not converge !!!!!!!!!!!!!!!!!!!!!! Exit the program!" << std::endl;
		  pcout << "    --- error = " << pre_error <<"				:gamma = " << gamma<< std::endl;
//		  std::exit(1);
	  }
	  system_matrix_tmp.clear();
	  system_rhs_tmp.clear();
//	  count++;
  }

  template<int dim>
  void HeatEquation<dim>::solve_system()
  {
	  pcout << std::endl << "***Time step " << timestep_number << " at t=" << time
			  << std::endl;
	  pcout << "===========================================" << std::endl
                    << "Number of active cells: " << triangulation.n_active_cells()  << std::endl
					<< "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;

//	  Vector<double> Temp_n_k = old_solution;		//Tn,k = Tn-1  initial guess
	  Vector<double> initial_cell_material = old_cell_material;
//	  assemble_system(/*Temp_n_k, */initial_cell_material);
//	  LA::MPI::SparseMatrix 		system_matrix_tmp;
//	  system_matrix_tmp.reinit (locally_owned_dofs,
//	                          locally_owned_dofs,
//	                          sparsity_pattern,
//	                          mpi_communicator);
//	  system_matrix_tmp.copy_from(system_matrix);
//	  LA::MPI::Vector system_rhs_tmp = system_rhs;
//	  system_matrix.copy_from(system_matrix_tmp);
//	  system_rhs = system_rhs_tmp;
//	  update_assemble_system(Temp_n_k, initial_cell_material);

	  // assemble --- start
	  if(true)
	  {
		  TimerOutput::Scope timer_section(computing_timer, "Assembling");

		  hp::FEValues<dim> hp_fe_values (fe_collection, quadrature_collection,
				  	  	  	  	  	  	  	  	  update_values | update_gradients |
												  update_quadrature_points | update_JxW_values);
		  // Finite element evaluated in quadrature points of the faces of a cell.
		  hp::FEFaceValues<dim> hp_fe_face_values(fe_collection, face_quadrature_collection,
				  	  	  	  	  	  	  	  	  update_values | update_quadrature_points |
												  update_quadrature_points | update_JxW_values);

		  system_matrix = 0;
		  system_rhs = 0;

		  const unsigned int           n_q_points    = quadrature_collection[0].size();		// n_q_points = 8
		  const unsigned int 			n_face_q_points = face_quadrature_collection[0].size(); // quadrature points on faces = 4
		  const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell; // dofs_per_active_cell = 8

		  FullMatrix<double> 			cell_matrix(dofs_per_active_cell, dofs_per_active_cell),
				  	  	  	  	  	  	  	  	  	  	  cell_mass_matrix(dofs_per_active_cell, dofs_per_active_cell),
														  cell_laplace_matrix(dofs_per_active_cell, dofs_per_active_cell);
		  FullMatrix<double>			    face_cell_mass_matrix (dofs_per_active_cell, dofs_per_active_cell);
		  Vector<double>	 		    	cell_rhs(dofs_per_active_cell);
		  Vector<double>      	        	face_cell_rhs (dofs_per_active_cell);

		  std::vector<types::global_dof_index> local_dof_indices (dofs_per_active_cell);

		  std::string old_source_type = source_type;
		  if (source_type == "Line" && time - time_step*1.5 < segment_start_time )
		  {// the first time step of current segment. change the previous source type to Point if swith from point to line, or Line if swith from line to point
			  {// heat source for previous segment is point
				  old_source_type = "Point";
				  std::cout<<"change to Point"<<std::endl;
			  }
		  }

//		  EquationData::InputHeatFlux<dim> rhs_function(parameters,
//			  source_type, scan_velocity, segment_start_time, segment_end_time, segment_length, orientation, segment_start_point, segment_end_point);
//		  std::vector<double>  rhs_old_values (n_q_points);//, rhs_values (n_q_points);

//		  std::vector<double> 		cell_old_solu_vectors(n_q_points);//, cell_solu_vectors(n_q_points);
		  Vector<double> 			cell_old_solu_Vectors(n_q_points);//, cell_solu_Vectors(n_q_points);

		  double     /*specific_heat_old, */specific_heat_Tnk,
		  	  	  	  	  conductivity_old, conductivity_Tnk,
						  convectivity_old, convectivity_Tnk,
						  emissivity_old, emissivity_Tnk;

		  // obtain the center point of laser beam and scaning orientation, only execute once. Later on, it will be needed in the stage of refine_mesh
//		  laserCenter = compute_laser_center(source_type, scan_velocity, segment_start_time, segment_end_time,
//	    												segment_start_point, segment_end_point, time - time_step, orientation); // calculate the center of laser
//		  double a = parameters.w, b = 1*parameters.w, c = 1*parameters.w; // three axis radius of the ellipsoid

		  MaterialData<dim> material_data (parameters);
		  double melt_point = parameters.melt_point;// = 1650 + 273;
		  double density = parameters.density;
		  const double Stephan_Boltzmann = 5.67e-8;
		  unsigned int cnt_cells (0), active_cnt_cells(0);

		  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  cell = dof_handler.begin_active(),
					  endc = dof_handler.end();
		  for (; cell!=endc; ++cell, ++cnt_cells)
		  {
			  if (cell->is_locally_owned())
			  {
				  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
				  if (dofs_per_cell != 0)
				  {
					  active_cnt_cells++;
					  cell_matrix = 0;
					  cell_mass_matrix = 0;
					  cell_laplace_matrix = 0;
					  cell_rhs = 0;

//					  hp_fe_values.reinit(cell);
//					  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
//					  if (fabs(time - layer_end_time - time_step) < 1e-7)
//					  {
//						  rhs_function.set_time(time - time_step);
//						  rhs_function.value_list(fe_values.get_quadrature_points(), rhs_old_values);		// Qn-1 ---> rhs_old_values
//					  }

//					  Vector<double> 			cell_old_solu_Vectors_tmp(n_q_points);
					  cell->get_dof_values(old_solution, cell_old_solu_Vectors); // get the local dof values at each node
//					  fe_values.get_function_values(old_solution, cell_old_solu_vectors);		// Tn-1  ---> cell_old_solu_vectors
//			  fe_values.get_function_values(Temp_n_k, cell_solu_vectors);	// Tnk  ---> cell_solu_vectors

					  double avg_old_tmp = 0., avg_tmp = 0.;
					  double num_melt_point_old = 0, num_melt_point = 0;

					  for (unsigned int t = 0; t < n_q_points; t++)
					  {
//						  cell_old_solu_Vectors[t] = cell_old_solu_vectors[t];
						  avg_old_tmp += cell_old_solu_Vectors[t];//cell_old_solu_vectors[t];
						  if (cell_old_solu_Vectors[t] >= melt_point)
							  num_melt_point_old++;
					  }
					  avg_old_tmp /= n_q_points;
					  avg_tmp = avg_old_tmp;
					  num_melt_point_old /= n_q_points;
					  num_melt_point = num_melt_point_old;

					  unsigned int old_material_id = old_cell_material[cnt_cells];
//					  specific_heat_old = material_data.get_specific_heat (avg_old_tmp, old_material_id);
					  conductivity_old = material_data.get_conductivity (avg_old_tmp, old_material_id);
					  convectivity_old = material_data.get_convectivity (avg_old_tmp, old_material_id);
					  emissivity_old = material_data.get_emissivity (avg_old_tmp, old_material_id);

//					  Point<dim> centerPt = cell->center();
//					  {// only consider the cell under the part height (activated cell)
//						  double local_x = (centerPt[0] - laserCenter[0])*cos(orientation) + (centerPt[1] - laserCenter[1])*sin(orientation),
//								  	  local_y = -(centerPt[0] - laserCenter[0])*sin(orientation) + (centerPt[1] - laserCenter[1])*cos(orientation);
//						  if (local_x > 0) // point in local axis lie in back of the laser heat. In this circumstance, elongate the long radius of the ellipsoid
//							  a = 0.25*parameters.w;
//						  else
//							  a = parameters.w;
//						  double distance = pow(local_x/(1*a), 2) + pow(local_y/(1*b), 2) +
//		    	    							  pow((centerPt[2] - laserCenter[2])/(1*c), 2) - 1;
//						  if (distance <= 0)
//						  {// center point of current active cell fall into the defined surface
//							  cell->set_material_id(0); // set material id to solid()
//						  }
//					  }
					  if (cell->material_id() == 1)
						  cell->set_material_id(0);
					  if (num_melt_point > 5/8)//(avg_tmp >= melt_point)
						  cell->set_material_id(1);
					  unsigned int material_id = cell->material_id();
					  cell_material[cnt_cells] = material_id;
					  initial_cell_material[cnt_cells] = material_id;
					  specific_heat_Tnk = material_data.get_specific_heat (avg_tmp, material_id);
					  conductivity_Tnk = material_data.get_conductivity (avg_tmp, material_id);
					  convectivity_Tnk = material_data.get_convectivity (avg_tmp, material_id);
					  emissivity_Tnk = material_data.get_emissivity (avg_tmp, material_id);

//					  if (fabs(time - layer_end_time - time_step) < 1e-7)
//						  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
//						  {
//							  for (unsigned int i=0; i<dofs_per_cell; ++i)
//							  {
//								  cell_rhs(i) += (0 + rhs_old_values [q_point]*(1-theta)
//														  )*time_step*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point);	// cell_rhs = Tau_n*theta*Fn+Tau_n*(1-theta)*Fn-1
//							  }
//						  }

					  cell_mass_matrix = cell_mass_matrix_list[cnt_cells];
					  cell_laplace_matrix = cell_laplace_matrix_list[cnt_cells];

					  double coeff_cell_mass_matrix = density*specific_heat_Tnk,
								  coeff_cell_laplace_matrix = conductivity_Tnk*time_step*theta;
					  cell_matrix.add(coeff_cell_mass_matrix, cell_mass_matrix, coeff_cell_laplace_matrix, cell_laplace_matrix);
						  // cell_matrix = M(Tn) + Tau_n*theta*K(Tn)

		//			  double coeff_cell_mass_matrix_old = density*specific_heat_old;
					  double coeff_cell_laplace_matrix_old = -time_step*(1-theta)*conductivity_old;
					  Vector<double> temp_mass(n_q_points);
					  Vector<double> temp_laplace(n_q_points);
					  cell_mass_matrix.vmult(temp_mass, cell_old_solu_Vectors);
					  cell_laplace_matrix.vmult(temp_laplace, cell_old_solu_Vectors);
					  cell_rhs.add(coeff_cell_mass_matrix, temp_mass, coeff_cell_laplace_matrix_old, temp_laplace);
						  // cell_rhs += M(Tn)*Tn-1 - Tau_n*(1-theta)*K(Tn-1)*Tn-1

					  if (layer_id > 5)		//(centerPt[dim - 1] > thickness*(layer_id - 1))
					  {
						  // Apply Boundary conditions
						  //
						  //
						  face_cell_mass_matrix = 0;
						  face_cell_rhs = 0;

						  ///---temperary add--start, during the cooling process, improve the convection coefficient
//						  convectivity_Tnk += 50;
//						  convectivity_old += 50;
//						  if (time > 150 && time < 470)
//						  {
//							  convectivity_Tnk += 100+(time-150)/8.;
//							  convectivity_old += 100+(time-150)/8.;;
//						  }
//						  if (time >= 470)
//						  {
//							  convectivity_Tnk += 200;
//							  convectivity_old += 200;;
//						  }
//						  convectivity_Tnk += 50;
//						  convectivity_old += 50;
//						  if (time < 73)
//						  {
//							  convectivity_Tnk =0;
//							  convectivity_old =0;
//							  emissivity_Tnk = 0;
//							  emissivity_old = 0;
//						  }
//						  else //if (time >= 470)
//						  {
//							  convectivity_Tnk += 400;
//							  convectivity_old += 400;;
//						  }
						  convectivity_Tnk -= time*4./395;
						  convectivity_old -= time*4./395;
						  ///---temperary add--end

						  double coeff_face_cell_rhs_conv = time_step*Tamb*(theta*convectivity_Tnk + (1-theta)*convectivity_old);
						  double coeff_face_cell_rhs_rad =time_step*Tamb*Stephan_Boltzmann*(theta*emissivity_Tnk*(avg_tmp + Tamb)*(avg_tmp*avg_tmp + Tamb*Tamb)
																								  +
																								  (1-theta)*emissivity_old*(avg_old_tmp + Tamb)*(avg_old_tmp*avg_old_tmp + Tamb*Tamb));
						  double coeff_face_cell_rhs = coeff_face_cell_rhs_conv + coeff_face_cell_rhs_rad;

						  double coeff_face_cell_matrix_conv = convectivity_Tnk*time_step*theta;
						  double coeff_face_cell_matrix_rad = emissivity_Tnk*Stephan_Boltzmann*(avg_tmp + Tamb) // avg_tmp should be replaced with Tnk
																								*(avg_tmp*avg_tmp + Tamb*Tamb)*time_step*theta;
						  double coeff_face_cell_matrix = coeff_face_cell_matrix_conv + coeff_face_cell_matrix_rad;

						  double coeff_face_cell_rhs_mass_conv = -time_step*(1-theta)*convectivity_old;
						  double coeff_face_cell_rhs_mass_rad = - time_step*(1-theta)*emissivity_old*Stephan_Boltzmann*(avg_old_tmp + Tamb)
																							  *(avg_old_tmp*avg_old_tmp + Tamb*Tamb);
						  double coeff_face_cell_rhs_mass = coeff_face_cell_rhs_mass_conv + coeff_face_cell_rhs_mass_rad;

						  for (unsigned int face_number=0; face_number <
													  GeometryInfo<dim>::faces_per_cell; ++face_number)
						  {
							  // Tests to select the cell faces which belong to the convection and radiation boundary
							  if (cell->face(face_number)->at_boundary()
											  && (cell->face(face_number)->boundary_id() == 1 || (cell->face(face_number)->boundary_id() == 3)))
							  {// convection boundary & radiation boundary
								  // Term to be added in the RHS
								  hp_fe_face_values.reinit (cell, face_number);
								  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();

								  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
								  {
									  // Computation and storage of the value of the shape functions on boundary face integration points
									  for (unsigned int i=0; i<dofs_per_cell; ++i)
									  {
										  for (unsigned int j=0; j<dofs_per_cell; ++j)
										  {
											  face_cell_mass_matrix(i, j) += (fe_face_values.shape_value(i, q_point)*fe_face_values.shape_value(j, q_point))*fe_face_values.JxW(q_point);
										  }
										  face_cell_rhs(i) += fe_face_values.shape_value(i, q_point)*fe_face_values.JxW(q_point);
										  // face_cell_rhs = Tau_n*theta*Tamb*Vconv_n + Tau_n*(1-theta)*Tamb*Vconv_n-1 +
										  //	  						Tau_n*theta*Tamb*Vrad_n + Tau_n*(1-theta)*Tamb*Vrad_n-1
									  }
								  }

								  cell_matrix.add(coeff_face_cell_matrix, face_cell_mass_matrix);

								  Vector<double> temp_mass_convection(dofs_per_cell);
								  face_cell_mass_matrix.vmult(temp_mass_convection, cell_old_solu_Vectors);
								  cell_rhs.add(coeff_face_cell_rhs_mass, temp_mass_convection);
								  cell_rhs.add(coeff_face_cell_rhs, face_cell_rhs);
							  }
						  }
					  }
					  cell->get_dof_indices (local_dof_indices);
					  constraints.distribute_local_to_global (cell_matrix, cell_rhs,
							  	  	  	  	  	  	  	  	  local_dof_indices,
															  system_matrix, system_rhs); // this doesn't work for hp elements, don't know why
				  }
			  }
		  }
		  system_matrix.compress(VectorOperation::add);
		  system_rhs.compress(VectorOperation::add);


	  }
	  // assemble --- end

	  solve_time_step();		// solution -> Tn*
  }

  template <int dim>
  bool HeatEquation<dim>::cell_is_in_metal_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
	  double limit = 0;
//	  if (time < layer_end_time + idle_time)
//		  limit = layer_id*thickness; // this is the part_height
//	  else
//		  limit = (1+layer_id)*thickness;

	  double current_max_limit = std::max(part_height, limit) - 0.05e-3/5;
	  Point<dim> centPt = cell->center();
	  return centPt[dim - 1] < current_max_limit;//in_metal;
  }

  template <int dim>
  bool HeatEquation<dim>::cell_is_in_void_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
	  double limit = 0;
//	  if (time < layer_end_time + idle_time) //segment_end_time  //floor(current_max_limit*1e6)*1e-6;
//		  limit = layer_id*thickness; // this is the part_height
//	  else
//		  limit = (1+layer_id)*thickness;

	  double current_max_limit = std::max(part_height, limit) - 0.05e-3/5;
	  Point<dim> centPt = cell->center();
	  return centPt[dim - 1] > current_max_limit;//in_void;
  }

  template <int dim>
  void HeatEquation<dim>::set_active_fe_indices ()
  {
	  // iteration over all the cells of the mesh
	  typename hp::DoFHandler<dim>::active_cell_iterator
	  			  cell_me = dof_handler_disp.begin_active();
	  for (typename hp::DoFHandler<dim>::active_cell_iterator
			  cell = dof_handler.begin_active();
			  cell != dof_handler.end(); ++cell, ++cell_me)
	  {
		  if (cell->is_locally_owned())
		  {
			  // Lagrange element if the cell is in the metal domain
			  if (cell_is_in_metal_domain(cell))
			  {
				  cell->set_active_fe_index (0);
				  cell_me->set_active_fe_index (0);
			  }
			  // Zero element if the cell is in the void domain
			  else if (cell_is_in_void_domain(cell))
			  {
				  cell->set_active_fe_index (1);
				  cell_me->set_active_fe_index (1);
			  }
			  // Throw an error if none of the two cases above is encountered
			  else
				  Assert (false, ExcNotImplemented());
		  }
	  }
  }

  int last_point_timestep_number = 0;
  template<int dim>
  void HeatEquation<dim>::output_results()
  {
	  TimerOutput::Scope timer_section(computing_timer, "Graphical output");
//	  if(timestep_number<358)
//		  return;

//	  if(timestep_number%100 == 0)
//	  {
//		  DataOut<dim,hp::DoFHandler<dim> > data_out;
//		  data_out.attach_dof_handler(dof_handler);
//		  data_out.add_data_vector(solution, "U");
//		  data_out.add_data_vector (FE_Type, "FE_Type");
//		  data_out.add_data_vector (cell_material, "cell_material");
//
//		  Vector<float> subdomain (dof_handler.get_triangulation().n_active_cells());
//		  for (unsigned int i=0; i<subdomain.size(); ++i)
//			  subdomain(i) = dof_handler.get_triangulation().locally_owned_subdomain();
//		  data_out.add_data_vector (subdomain, "Subdomain");
//
//		  data_out.build_patches();
//
//		  std::string filename = "solution-"
//										 + Utilities::int_to_string(timestep_number, 3) +
//										 "." +
//										 Utilities::int_to_string
//										 (dof_handler.get_triangulation().locally_owned_subdomain(), 4);
//		  filename = output_heat_dir + filename;
//		  std::ofstream output((filename + ".vtu").c_str());
//		  data_out.write_vtu(output);
//
//		  if (this_mpi_process == 0)
//		  {
//			  std::vector<std::string> filenames;
//			  for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
//				  filenames.push_back ("solution-" +
//														  Utilities::int_to_string (timestep_number, 3) +
//														  "." +
//														  Utilities::int_to_string (i, 4) +
//														  ".vtu");
//			  std::ofstream master_output ((output_heat_dir + "solution-" +
//														  Utilities::int_to_string (timestep_number, 3) +
//														  ".pvtu").c_str());
//			  data_out.write_pvtu_record (master_output, filenames);  // defaulty, write_pvtu_record() function will search the filenames in the current folder.
//		  }
//	  }

//	  if(timestep_number%5 == 0 || time > layer_end_time)
	  {
		  const Point<dim> point_A(0.0107429, 0, 0); //(-0.00937143, 0, 0); //(-0.0114286, 0, 0);  //(-0.0121143, 0, 0);
		  Vector<double>   disp_A(1);		//(dim);
		  Vector<double>   copy_solution(solution);

//		  Evaluation::PointValuesEvaluation<dim> point_values_evaluation(point_A);
//		  point_values_evaluation.compute (dof_handler, copy_solution, disp_A);

	//	  if(disp_A[0] < 1e10)
		  {
			  if (this_mpi_process == 0)
			  {
//				  table_results.set_auto_fill_mode(true);
				  table_results.add_value("time", time);
				  table_results.set_precision("time", 7);
			  }
			  table_results.add_value("u_A", disp_A(0));

			  std::string filename_2 = "Results_A_" + Utilities::int_to_string(dof_handler.get_triangulation().locally_owned_subdomain(), 4);
			  filename_2 = output_heat_dir + filename_2;
			  std::ofstream output_txt((filename_2 + ".txt").c_str());
			  table_results.write_text(output_txt);
		  }
//		  const Point<dim> point_B(0.0114286, 0, 0); //(-0.0107429, 0, 0);
//		  Vector<double>   disp_B(1);		//(dim);
//
//		  Evaluation::PointValuesEvaluation<dim> point_values_evaluation_B(point_B);
//		  point_values_evaluation_B.compute (dof_handler, copy_solution, disp_B);
//	//	  if(disp_B[0] < 1e10)
//		  {
//	//	  	  table_results_2.add_value("time", time);
//	//	  	  table_results_2.set_precision("time", 7);
//			  table_results_2.add_value("u_B", disp_B(0));
//
//			  std::string filename_2 = "Results_B_" + Utilities::int_to_string(dof_handler.get_triangulation().locally_owned_subdomain(), 4);
//			  filename_2 = output_heat_dir + filename_2;
//			  std::ofstream output_txt_2((filename_2 + ".txt").c_str());
//			  table_results_2.write_text(output_txt_2);
//		  }
//
//		  const Point<dim> point_C(0.0121143, 0, 0);//(pt_pos, 0, 0); //(0.0121143, 0, 0);
//		  Vector<double>   disp_C(1);		//(dim);
//
//		  Evaluation::PointValuesEvaluation<dim> point_values_evaluation_C(point_C);
//		  point_values_evaluation_C.compute (dof_handler, copy_solution, disp_C);
//		  {
//			  table_results_3.add_value("u_C", disp_C(0));
//
//			  std::string filename_2 = "Results_C_" + Utilities::int_to_string(dof_handler.get_triangulation().locally_owned_subdomain(), 4);
//			  filename_2 = output_heat_dir + filename_2;
//			  std::ofstream output_txt((filename_2 + ".txt").c_str());
//			  table_results_3.write_text(output_txt);
//		  }
	  }
  }

  template <int dim>
  void HeatEquation<dim>::refine_mesh (const int min_grid_level,
                                       const int max_grid_level)
  {
	  TimerOutput::Scope timer_section(computing_timer, "refine_mesh");

	  std::vector<Point<dim>> bounding_4_points = compute_vertexes_of_bounding_box(segment_start_point, segment_end_point, 2*parameters.w, 6*parameters.w);

	  double limit = part_height;

	  for (int step = min_grid_level; step <= max_grid_level; ++step)
	  {
		  pcout<<"refine_mesh --- step_"<<step<<std::endl;
		  unsigned int active_cnt = 0, cnt_refine_cell = 0, cnt_coarsen_cell = 0;

		  for (typename Triangulation<dim>::active_cell_iterator
				  	  cell = triangulation.begin_active();
				  	  cell != triangulation.end(); ++cell, active_cnt++)
		  {
			  Point<dim> centerPt = cell->center();
			  if (centerPt[2] <= limit)
			  {// only consider the cell under the part height (activated cell)
				  if (Is_Point_in_rectangle(bounding_4_points, centerPt) && centerPt[2] >= part_height - 1.5*parameters.w)
				  {// distance <= 0 && centerPt[2] >= depth, center point of current active cell fall in of the refine surface
					  if (cell->level() < max_grid_level)
					  {
						  cell->set_refine_flag();
						  cnt_refine_cell++;
					  }
				  }
				  else
				  {// center point of current active cell fall out of the refine surface
					  if (centerPt[2] > 0) // part domain
					  {
						  if (/*min_grid_level <= cell->level() && */cell->level() <= max_grid_level)
						  {// refine level 5 will be coarsen
							  cell->set_coarsen_flag();
							  cnt_coarsen_cell++;

							  Point<dim> test_direction;
							  for(unsigned int coord = 0; coord < dim; coord++)
								  test_direction[coord]= centerPt[coord] - segment_start_point[coord];
							  double angle = calculate_angle_btw_2_vectors(print_direction, test_direction);

//							  if ((cell->level() == min_grid_level && cell_material(cnt_cells) != 2) || (cell->level() == (max_grid_level - 1) && cell_material(cnt_cells) == 2))
							  if ((cell->level() == min_grid_level && angle > 90) || (cell->level() == (max_grid_level - 1) && angle <= 90))
							  {
								  // only consider the cell under the part height (activated cell)
								  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
								  {
									  const Point<dim> face_center = cell->face(f)->center();
									  if (fabs(face_center[2] - limit) < 1e-6)
									  {
										  cell->clear_coarsen_flag ();
										  break;
									  }
								  }
							  }
						  }
					  }
					  else	//centerPt[2] <= 0, substrate domain
					  {
						  cell->set_coarsen_flag();
						  cnt_coarsen_cell++;
					  }
				  }
			  }
		  }
		  pcout<<"------------------ step_"<<step<<", active_cnt = "<<active_cnt<<std::endl;
		  cout<<"------------------ step_"<<step<<", cnt refine cell = "<<cnt_refine_cell<<", coarsen cell = "<<cnt_coarsen_cell<<", in processor:"<<this_mpi_process<<std::endl;
//		  if (cnt_refine_cell == 0)// && cnt_coarsen_cell == 0)
//			  continue;

		  if (triangulation.n_levels() > (unsigned int)max_grid_level) // 5
			  for (typename Triangulation<dim>::active_cell_iterator
					  cell = triangulation.begin_active(max_grid_level);
					  cell != triangulation.end(); ++cell)
				  cell->clear_refine_flag ();
		// Considering the bug in deal.II library all coarsen flags are removed
		// Otherwise only the coarsening flag of the cells which are at the minimal level of refinement and non-powder material type would be cleared

//		// close temperary
//		for (typename Triangulation<dim>::active_cell_iterator
//	         cell = triangulation.begin_active(min_grid_level);//(min_grid_level-3);
//	         cell != triangulation.end_active(min_grid_level); ++cell)//(min_grid_level-3); ++cell)
//	    {
//	    		cell->clear_coarsen_flag ();
//	    }

//		  for (typename Triangulation<dim>::active_cell_iterator
//					cell = triangulation.begin_active(max_grid_level - 1);
//					cell != triangulation.end_active(max_grid_level - 1); ++cell)
//		  {
//			  Point<dim> centerPt = cell->center();
//			  if (centerPt[2] <= limit)
//			  {// only consider the cell under the part height (activated cell)
//				  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//				  {
//					  const Point<dim> face_center = cell->face(f)->center();
//					  if (fabs(face_center[2] - limit) < 1e-6)
//					  {
//						  cell->clear_coarsen_flag ();
//						  break;
//					  }
//				  }
//			  }
//		  }


		  unsigned int cell_for_coarsen(0), cell_total(0), cell_total_childs(0), cell_sameid(0), cell_differid(0);
		  for (typename Triangulation<dim>::cell_iterator
				  	  cell = triangulation.last();//end(max_grid_level);
				  	  cell != triangulation.begin(); cell--)//, old_cell_iter--)//, ++total_cells)
		  {
			  unsigned int no_of_all_children = cell->number_of_children(); // get the number of all child cells
			  if (!cell->active() && no_of_all_children == 4*(dim - 1))
			  {	// current cell has children (non-active cell) and only has 8 children cell in total
				  cell_total++;
				  Vector<double> cell_material_id;//, old_cell_material_id;
				  cell_material_id.reinit(4*(dim - 1)); //	old_cell_material_id.reinit(4*(dim - 1));
				  double avg_material_id = 0;//, avg_old_material_id = 0;
				  unsigned int powder_count = 0, liquid_count = 0, solid_count = 0;
				  for (unsigned int i = 0; i < 4*(dim - 1); i++)
				  {
					  cell_total_childs++;
					  typename Triangulation<dim>::active_cell_iterator child_cell = cell->child(i);//,
//																								  old_child_cell = old_cell_iter->child(i);
					  cell_material_id[i] = child_cell->material_id();
					  avg_material_id += child_cell->material_id();
					  if (child_cell->material_id() == 2)
						  powder_count++;
					  else if (child_cell->material_id() == 1)
						  liquid_count++;
					  else if (child_cell->material_id() == 0)
						  solid_count++;
					// for old material on old mesh
//					old_cell_material_id[i] = old_child_cell->material_id();
//					avg_old_material_id += old_child_cell->material_id();
//					if (old_child_cell->material_id() == 2)
//						old_powder_count++;
//					else if (old_child_cell->material_id() == 1)
//						old_liquid_count++;
//					else if (old_child_cell->material_id() == 0)
//						old_solid_count++;
				  }

				  if (fabs(avg_material_id/(4*(dim - 1)) - cell_material_id[0]) < 1e-3)
				  {
					  cell_sameid++;
					  cell->set_material_id(cell_material_id[0]);
				  }
				  else
				  {
					  cell_differid++;
					  if (powder_count > 4)
						  cell->set_material_id(2); // set material id as powder.
					  else if (liquid_count > 4)
						  cell->set_material_id(1); // set material id as liquid.
					  else if (solid_count > 4)
						  cell->set_material_id(0); // set material id as solid.
					  else
						  for (unsigned int i = 0; i < 4*(dim - 1); i++)
						  {
							  typename Triangulation<dim>::active_cell_iterator child_cell = cell->child(i);//,
//																				 old_child_cell = old_cell_iter->child(i);
							  if (child_cell->active())
							  {
								  cell_for_coarsen++;
//								  child_cell->clear_coarsen_flag(); // close temperary! as it will crash triangulation.execute_refine_and_coarsen().
							  }
						  }
	//    			cell->set_material_id(0/*1*/); // set material id as solid.
				  }
//				// for old material type
//				if (fabs(avg_old_material_id/(4*(dim - 1)) - old_cell_material_id[0]) < 1e-3)
//				{
//					old_cell_iter->set_material_id(old_cell_material_id[0]);
//				}
//				else
//				{
//					if (old_powder_count > 4)
//						old_cell_iter->set_material_id(2); // set material id as powder.
//					else if (old_liquid_count > 4)
//						old_cell_iter->set_material_id(1); // set material id as liquid.
//					else if (old_solid_count > 4)
//						old_cell_iter->set_material_id(0); // set material id as solid.
//	//    			else
//	//    				for (unsigned int i = 0; i < 4*(dim - 1); i++)
//	//    				{
//	//    					typename Triangulation<dim>::active_cell_iterator old_child_cell = old_cell_iter->child(i);
//	//    					if (old_child_cell->active())
//	//    						old_child_cell->clear_coarsen_flag();
//	//    				}
//				}
			  }
		  }
		  unsigned int cnt_cells (0);
		  for (typename Triangulation<dim>::active_cell_iterator
				  	  cell = triangulation.begin_active();
				  	  cell != triangulation.end(); ++cell, ++cnt_cells)
		  { // reset the cell_material
			  cell_material[cnt_cells] = cell->material_id();
		  }


//		  std::vector<bool> vector_refine_flags(triangulation.n_active_cells()),
//				  	  	  	vector_coarsen_flags(triangulation.n_active_cells());
//		  triangulation.save_refine_flags(vector_refine_flags);
//		  triangulation.save_coarsen_flags(vector_coarsen_flags);
//		  double flags_norm_refine = 0, flags_norm_coarsen = 0;
//		  for(unsigned int i = 0; i< vector_refine_flags.size(); i++)
//		  {
//			  if(vector_refine_flags[i])
//				  flags_norm_refine += 1;
//			  if(vector_coarsen_flags[i])
//				  flags_norm_coarsen += 1;
//		  }
//		  cout<<"   --- flags_norm_refine = "<< flags_norm_refine<<", flags_norm_coarsen = "<< flags_norm_coarsen<<endl;

//		  cout <<"processor id: "<< this_mpi_process<<", ####### cell_for_coarsen: "<<cell_for_coarsen<<", cell_total = "<<cell_total
//				<<", cell_total_childs = "<<cell_total_childs
//				<<", cell_differid = "<<cell_differid
//				<<", cell_sameid = "<<cell_sameid<<endl;


		  //--- for mechanical analysis ---history variables on quadrature points-start
//		  if(thermal_mechanical_flg)
//		  {
				  // ---------------------------------------------------------------
				  // Make a field variable for history varibales to be able to transfer the data to the quadrature points of the new mesh
				FE_DGQ<dim> history_fe (1);
				DoFHandler<dim> history_dof_handler (triangulation);
				history_dof_handler.distribute_dofs (history_fe);

				std::vector< std::vector< Vector<double> > >
									history_stress_field (dim, std::vector< Vector<double> >(dim)),		// pre_stress
									local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
									local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim));
				std::vector< std::vector< Vector<double> > >
									history_strain_field (dim, std::vector< Vector<double> >(dim)),		// pre_plastic_strain
									local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
									local_history_strain_fe_values (dim, std::vector< Vector<double> >(dim));
				Vector< double>
									history_epstrain_field (history_dof_handler.n_dofs()),		// effective_plastic_strain (double)
									local_history_epstrain_values_at_qpoints (quadrature_collection_disp[0].size()), // quadrature_collection_disp[0].size()=8
									local_history_epstrain_fe_values (history_fe.dofs_per_cell);//history_fe.dofs_per_cell = 8

				IndexSet history_locally_owned_dofs = history_dof_handler.locally_owned_dofs ();
				LA::MPI::Vector mpi_history_epstrain_field(history_locally_owned_dofs, mpi_communicator);

				std::vector<LA::MPI::Vector> mpi_history_field_row(dim);
				std::vector< std::vector<LA::MPI::Vector> > mpi_history_stress_field(dim, mpi_history_field_row);//std::vector< Vector<double> >(dim, mpi_history_epstrain_field));
				std::vector< std::vector<LA::MPI::Vector> > mpi_history_strain_field(dim, mpi_history_field_row);//(dim);//(dim, std::vector< Vector<double> >(dim));

				for (unsigned int i=0; i<dim; i++)
					for (unsigned int j=0; j<dim; j++)
					{
						mpi_history_stress_field[i][j].reinit(history_locally_owned_dofs, mpi_communicator);
						mpi_history_strain_field[i][j].reinit(history_locally_owned_dofs, mpi_communicator);
					}

				for (unsigned int i=0; i<dim; i++)
					for (unsigned int j=0; j<dim; j++)
					{
						history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
						local_history_stress_values_at_qpoints[i][j].reinit(quadrature_collection_disp[0].size());
						local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);

						history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
						local_history_strain_values_at_qpoints[i][j].reinit(quadrature_collection_disp[0].size());
						local_history_strain_fe_values[i][j].reinit(history_fe.dofs_per_cell);
					}

				FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
																	quadrature_collection_disp[0].size());
				FETools::compute_projection_from_quadrature_points_matrix(history_fe,
									quadrature_collection_disp[0], quadrature_collection_disp[0], qpoint_to_dof_matrix);

				typename hp::DoFHandler<dim>::active_cell_iterator
												cell_me = dof_handler_disp.begin_active(),
												endc_me = dof_handler_disp.end();
				typename DoFHandler<dim>::active_cell_iterator
												dg_cell = history_dof_handler.begin_active();
				for (; cell_me!=endc_me; ++cell_me, ++dg_cell)
					if (cell_me->is_locally_owned())
					{
						const unsigned int dofs_per_cell_disp = cell_me -> get_fe().dofs_per_cell;
						if (dofs_per_cell_disp != 0) // To skip the cell with FE = FE_Nothing because there is no support point there
						{
						PointHistory<dim> *local_quadrature_points_history
															= reinterpret_cast<PointHistory<dim> *>(cell_me->user_pointer());
						Assert (local_quadrature_points_history >=
							  &quadrature_point_history.front(), ExcInternalError());
						Assert (local_quadrature_points_history <
							  &quadrature_point_history.back(), ExcInternalError());
						for (unsigned int i=0; i<dim; i++)
							for (unsigned int j=0; j<dim; j++)
							{
								for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
								{
									local_history_stress_values_at_qpoints[i][j](q)
												  = local_quadrature_points_history[q].pre_stress[i][j];

									local_history_strain_values_at_qpoints[i][j](q)
												  = local_quadrature_points_history[q].pre_plastic_strain[i][j];
								}
								qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
														local_history_stress_values_at_qpoints[i][j]);
								dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
														history_stress_field[i][j]); //write the value of 'local_history_stress_fe_values' into history_stress_field.
								qpoint_to_dof_matrix.vmult (local_history_strain_fe_values[i][j],
														local_history_strain_values_at_qpoints[i][j]);
								dg_cell->set_dof_values (local_history_strain_fe_values[i][j],
														history_strain_field[i][j]);
							}
						// mapping effective plastic strain from quadrature points to Dof
						for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
						{
							local_history_epstrain_values_at_qpoints(q)
												  = local_quadrature_points_history[q].pre_effective_plastic_strain;
//							cout<<"local_history_epstrain(q)="<<local_history_epstrain_values_at_qpoints(q)<<endl;
						}
						qpoint_to_dof_matrix.vmult (local_history_epstrain_fe_values,
													local_history_epstrain_values_at_qpoints);
						dg_cell->set_dof_values (local_history_epstrain_fe_values,
													history_epstrain_field);
						}
					}
				// added-0606-start--update local to global
				for (unsigned int i=0; i<dim; i++)
					for (unsigned int j=0; j<dim; j++)
					{
						//stress
						mpi_history_stress_field[i][j] = history_stress_field[i][j];
						history_stress_field[i][j] = mpi_history_stress_field[i][j];
						//strain
						mpi_history_strain_field[i][j] = history_strain_field[i][j];
						history_strain_field[i][j] = mpi_history_strain_field[i][j];
					}
				// effective plastic strain
				mpi_history_epstrain_field = history_epstrain_field;
				history_epstrain_field = mpi_history_epstrain_field;
				// added-0606-end--update local to global
//		  }
		  //--- for mechanical analysis ---history variables on quadrature points-end

		  pcout<<"refine_mesh --- step_"<<step<<" ----- end!"<<std::endl;
		  triangulation.prepare_coarsening_and_refinement();

		  // material id transfer for material_id --- start
		  SolutionTransfer<dim, Vector<double>, DoFHandler<dim>>
//		  SolutionTransfer<dim, LA::MPI::Vector, DoFHandler<dim>>
		  	  	  	  	  	  	  	  	  material_field_transfer(material_dof_handler);
		  Vector< double> material_field (material_dof_handler.n_dofs());

		  unsigned int dofs_per_cell = history_fe_material.dofs_per_cell;

		  LA::MPI::Vector distributed_material_id_per_cell;

		  std::vector<bool> point_saved(material_dof_handler.n_dofs(), false);
		  distributed_material_id_per_cell.reinit(material_locally_owned_dofs, //material_locally_relevant_dofs,
  																		mpi_communicator);
		  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		  unsigned int cnt_cells_for_material(0);
//		  unsigned int active_cnt_cells_incurrent(0);
		  for (typename DoFHandler<dim>::active_cell_iterator
				  	  cell = material_dof_handler.begin_active();
				  	  cell!=material_dof_handler.end(); ++cell, ++cnt_cells_for_material)
		  {
			  if (cell->is_locally_owned())
			  {
				  cell->get_dof_indices (local_dof_indices);
				  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
					  if (point_saved[local_dof_indices[i]] == true)
					  {  /*pcout<<"find similar ----"<<endl;*/continue; }
					  distributed_material_id_per_cell(local_dof_indices[i]) = cell_material[cnt_cells_for_material];
					  point_saved[local_dof_indices[i]] = true;
				  }
//				  active_cnt_cells_incurrent++;
			  }
		  }
//		  cout<<"active_cnt_cells_incurrent: "<<active_cnt_cells_incurrent<<endl;

		  distributed_material_id_per_cell.compress (VectorOperation::insert);
		  material_field = distributed_material_id_per_cell;

//		  LA::MPI::Vector local_solution_material(material_locally_owned_dofs, material_locally_relevant_dofs,
//  		            											mpi_communicator);
//		  LA::MPI::Vector local_solution_material_no_ghost_vector(material_locally_owned_dofs, //locally_relevant_dofs,
//  		            											mpi_communicator);
//		  local_solution_material_no_ghost_vector = material_field;
//		  local_solution_material = local_solution_material_no_ghost_vector;
//		  material_field_transfer.prepare_for_coarsening_and_refinement(local_solution_material);
		  material_field_transfer.prepare_for_coarsening_and_refinement(material_field);
		  // material id transfer for material_id --- end


		  // solution transfer for heat (dof_handler)
		  SolutionTransfer<dim, Vector<double>, hp::DoFHandler<dim>>
//				solution_trans(dof_handler),
		  	  	  	  old_solution_trans(dof_handler);
//		solution_trans.prepare_for_coarsentimestep_numbering_and_refinement(solution);
		  old_solution_trans.prepare_for_coarsening_and_refinement(old_solution);

		  //--- for mechanical analysis ---solutionTransfer-start
//		  if(thermal_mechanical_flg)
//		  {
				SolutionTransfer<dim, Vector<double>, hp::DoFHandler<dim>>
						solution_trans_disp(dof_handler_disp);

				solution_trans_disp.prepare_for_coarsening_and_refinement(solution_disp);

				SolutionTransfer<dim, Vector<double>, DoFHandler<dim> >
								history_stress_field_transfer0(history_dof_handler),
								history_stress_field_transfer1(history_dof_handler),
								history_stress_field_transfer2(history_dof_handler);
				history_stress_field_transfer0.prepare_for_coarsening_and_refinement(history_stress_field[0]);
				if ( dim > 1)
				{
					history_stress_field_transfer1.prepare_for_coarsening_and_refinement(history_stress_field[1]);
				}
				if ( dim == 3)
				{
					history_stress_field_transfer2.prepare_for_coarsening_and_refinement(history_stress_field[2]);
				}

				SolutionTransfer<dim, Vector<double>, DoFHandler<dim> >
								history_strain_field_transfer0(history_dof_handler),
								history_strain_field_transfer1(history_dof_handler),
								history_strain_field_transfer2(history_dof_handler);
				history_strain_field_transfer0.prepare_for_coarsening_and_refinement(history_strain_field[0]);
				if ( dim > 1)
				{
					history_strain_field_transfer1.prepare_for_coarsening_and_refinement(history_strain_field[1]);
				}
				if ( dim == 3)
				{
					history_strain_field_transfer2.prepare_for_coarsening_and_refinement(history_strain_field[2]);
				}

				SolutionTransfer<dim, Vector<double>, DoFHandler<dim> >
								history_epstrain_field_transfer(history_dof_handler);
				history_epstrain_field_transfer.prepare_for_coarsening_and_refinement(history_epstrain_field);
//		  }
		  //--- for mechanical analysis ---solutionTransfer-end

		  pcout<<"refine_mesh --- triangulation.execute before_"<<step<<std::endl;
		  triangulation.execute_coarsening_and_refinement ();
		  pcout<<"refine_mesh --- triangulation.execute after_"<<step<<std::endl;

//		old_triangulation.execute_coarsening_and_refinement ();
//		  pcout<<" triangulation.size = "<< triangulation.n_active_cells()<<endl;//", old triangulation.size = "<< old_triangulation.n_active_cells()<<std::endl;

//		  if(thermal_mechanical_flg)
			  setup_quadrature_point_history (); // for mechanical analysis

		  // added in 2019-0207 --start
 		  set_active_fe_indices();
	 	  setup_system();
		  // added in 2019-0207 --end
		  // added in 2019-0527 --start
//	 	  if(thermal_mechanical_flg)
	 	  	  Vector<double> tmp_solution_disp = solution_disp;
	 		  setup_mech_system();
		  // added in 2019-0527 --end

		  /////////---------material id transfer --- start
//		  material_dof_handler.distribute_dofs (history_fe_material);
//
//		  material_locally_owned_dofs = material_dof_handler.locally_owned_dofs ();
//		  DoFTools::extract_locally_relevant_dofs (material_dof_handler,
//	   													 material_locally_relevant_dofs);

		  Vector<double> new_material_field(material_dof_handler.n_dofs());
		  material_field_transfer.interpolate(material_field, new_material_field);
//cout<<"before interpolate material_field.l2_norm()="<<material_field.l2_norm()<<", in CPU-"<<this_mpi_process<<endl;
		  material_field = new_material_field;
//cout<<"after interpolate material_field.l2_norm()="<<material_field.l2_norm()<<", in CPU-"<<this_mpi_process<<endl;
//		  cell_material.reinit(triangulation.n_active_cells());
		  cnt_cells_for_material = 0;//, active_cnt_cells(0);
		  for (typename DoFHandler<dim>::active_cell_iterator
	   	   				cell = material_dof_handler.begin_active();
	   	   				cell!=material_dof_handler.end(); ++cell, ++cnt_cells_for_material)
		  {
			  if (cell->is_locally_owned())
			  {
				  cell->get_dof_indices (local_dof_indices);
				  Vector<double> 			cell_material_type(dofs_per_cell);
				  cell->get_dof_values(material_field, cell_material_type);
				  unsigned int powder_count = 0, solid_count = 0, liquid_count = 0;
				  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
					  if(material_field(local_dof_indices[i]) == 2)
						  powder_count++;
					  else if(material_field(local_dof_indices[i]) == 1)
						  liquid_count++;
					  else if(material_field(local_dof_indices[i]) == 0)
						  solid_count++;
				  }
				  unsigned int material_type = 0;
				  if(powder_count > liquid_count && powder_count > solid_count)
					  material_type = 2;
				  else if(solid_count > liquid_count && solid_count > powder_count)
					  material_type = 0;
				  else if(liquid_count > powder_count && liquid_count > solid_count)
					  material_type = 1;
				  cell_material[cnt_cells_for_material] = material_type;
	   	//			distributed_material_id_per_cell[cnt_cells_for_material] = material_type;
				  cell->set_material_id(material_type);
			  }
		  }
		  /////////---------material id transfer --- end
		  pcout<<"refine_mesh --- test 1"<<std::endl;
//		  set_active_fe_indices();
//		  pcout<<"refine_mesh --- test 2"<<std::endl;

//		  dof_handler.distribute_dofs(fe_collection);
//		  locally_owned_dofs = dof_handler.locally_owned_dofs ();
//		  DoFTools::extract_locally_relevant_dofs (dof_handler,
//														 locally_relevant_dofs);

//		  dof_handler_disp.distribute_dofs (fe_collection_disp);
//		  pcout<<"refine_mesh --- test 3"<<std::endl;
		  // Vector to visualize the material type of each cell
//		  cell_material.reinit(triangulation.n_active_cells());
		  old_cell_material.reinit(triangulation.n_active_cells());
		  old_cell_material = cell_material;
		  // Vector to visualize the FE of each cell
//		  FE_Type.reinit(triangulation.n_active_cells());
//		  cnt_cells = 0;
//		  typename hp::DoFHandler<dim>::active_cell_iterator
//		  	  	  	  cell = dof_handler.begin_active(),
//					  endc = dof_handler.end();
//		  for (; cell!=endc; ++cell)//, ++old_cell)
//		  {
//			  unsigned int material_id = cell->material_id();
//			  old_cell_material[cnt_cells] = material_id;//old_material_id;

//			  unsigned int fe_index = cell->active_fe_index();
//			  FE_Type[cnt_cells] = fe_index;
//			  ++ cnt_cells;
//		  }
		  pcout<<", cnt cell in DOF = "<< cnt_cells <<std::endl;


		  // Heat Solution interpolation on the new Dof handler --- start
		  Vector<double> /*new_solution(dof_handler.n_dofs()), */new_old_solution(dof_handler.n_dofs());
	//    std::cout << "test 0" << std::endl;
//		  solution_trans.interpolate(solution, new_solution);
//		  pcout<<"refine_mesh --- new dof_handler.size() = "<<dof_handler.n_dofs()<<", old_solution.size() = "<<old_solution.size()<<std::endl;
		  old_solution_trans.interpolate(old_solution, new_old_solution);
	//    std::cout << "test 1" << std::endl;
		  solution.reinit(dof_handler.n_dofs());
		  old_solution.reinit(dof_handler.n_dofs());
//		  solution = new_solution;
		  old_solution = new_old_solution;
		  pcout<<"refine_mesh --- test 4"<<std::endl;
//		  constraints.clear();
//		  constraints.reinit(locally_relevant_dofs);
//		  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
//		  VectorTools::interpolate_boundary_values(dof_handler,
//				  	  	  	  	  	  	  	  	  	  21,
//													  EquationData::BoundaryValues<dim>(),
//													  constraints); // interpolate Dirichlet boundary condition
//
//		  constraints.close();
		  // Computation of the hanging nodes constraints
//		  constraints.distribute (solution);
		  constraints.distribute (old_solution);
		  // Heat Solution interpolation on the new Dof handler --- end
		  pcout<<"refine_mesh --- test 5"<<std::endl;


		  // Displacement solution interpolation on the new Dofhandler_disp -- mechanical analysis
//		  if(thermal_mechanical_flg)
//		  {
//		  	    pcout<<"refine_mesh --- new dof_handler_disp.size() = "<<dof_handler_disp.n_dofs()<<", tmp_solution_disp.size() = "<<tmp_solution_disp.size()<<std::endl;
				Vector<double> distributed_solution_disp(dof_handler_disp.n_dofs());
				pcout<<"refine_mesh --- test 6"<<std::endl;
				solution_trans_disp.interpolate(tmp_solution_disp/*solution_disp*/, distributed_solution_disp);

				cout<<"   $$$$ tmp_solution_disp.l2_norm() = "<<tmp_solution_disp.l2_norm()<<", distributed_solution_disp.l2_norm() = "<<distributed_solution_disp.l2_norm()<<endl;
				pcout<<"refine_mesh --- test 7"<<std::endl;
				solution_disp.reinit(dof_handler_disp.n_dofs());
				solution_disp = distributed_solution_disp;
//				std::cout << "hanging node test 4" << std::endl;
//				constraints_disp.clear();
//				DoFTools::make_hanging_node_constraints(dof_handler_disp, constraints_disp);
//				std::cout << "hanging node test 5" << std::endl;
//				constraints_disp.close();
//				compute_dirichlet_constraints();
			//    std::cout << "test 6" << std::endl;
				// Computation of the hanging nodes constraints
				constraints_dirichlet_and_hanging_nodes.distribute (solution_disp);
			//    std::cout << "test 7" << std::endl;
				// ---------------------------------------------------
				history_dof_handler.distribute_dofs (history_fe);
				// stress
				std::vector< std::vector< Vector<double> > >
								distributed_history_stress_field(dim, std::vector< Vector<double> >(dim));
				for (unsigned int i=0; i<dim; i++)
					for (unsigned int j=0; j<dim; j++)
					{
						distributed_history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
					}

				history_stress_field_transfer0.interpolate(history_stress_field[0], distributed_history_stress_field[0]);
				if ( dim > 1)
				{
					history_stress_field_transfer1.interpolate(history_stress_field[1], distributed_history_stress_field[1]);
				}
				if ( dim == 3)
				{
					history_stress_field_transfer2.interpolate(history_stress_field[2], distributed_history_stress_field[2]);
				}

				history_stress_field = distributed_history_stress_field;

				// plastic strain
				std::vector< std::vector< Vector<double> > >
							distributed_history_strain_field (dim, std::vector< Vector<double> >(dim));
				for (unsigned int i=0; i<dim; i++)
					for (unsigned int j=0; j<dim; j++)
					{
						distributed_history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
					}

				history_strain_field_transfer0.interpolate(history_strain_field[0], distributed_history_strain_field[0]);
				if ( dim > 1)
				{
					history_strain_field_transfer1.interpolate(history_strain_field[1], distributed_history_strain_field[1]);
				}
				if ( dim == 3)
				{
					history_strain_field_transfer2.interpolate(history_strain_field[2], distributed_history_strain_field[2]);
				}

				history_strain_field = distributed_history_strain_field;

				// effective plastic strain
				Vector<double>
							distributed_history_epstrain_field (history_dof_handler.n_dofs());
				history_epstrain_field_transfer.interpolate(history_epstrain_field, distributed_history_epstrain_field);

//				pcout<<"before history_epstrain_field.size() = "<<history_epstrain_field.size()<<endl;
//				history_epstrain_field.reinit(history_dof_handler.n_dofs());
//				pcout<<"after history_epstrain_field.size() = "<<history_epstrain_field.size()<<endl;
				history_epstrain_field = distributed_history_epstrain_field;
//				cout<<"history_epstrain_field.l2_norm = "<<history_epstrain_field.l2_norm()<<
//						",history_stress_field[0][1].l2_norm = "<<history_stress_field[0][1].l2_norm()<<
//						",history_strain_field[2][2].l2_norm = "<<history_strain_field[2][2].l2_norm()<<endl;

				// ---------------------------------------------------------------
				// Transfer the history data to the quadrature points of the new mesh
				// In a final step, we have to get the data back from the now
				// interpolated global field to the quadrature points on the
				// new mesh. The following code will do that:

				FullMatrix<double> dof_to_qpoint_matrix (quadrature_collection_disp[0].size(),
														 history_fe.dofs_per_cell);
				FETools::compute_interpolation_to_quadrature_points_matrix (history_fe,
														quadrature_collection_disp[0], dof_to_qpoint_matrix);

				cell_me = dof_handler_disp.begin_active();
				endc_me = dof_handler_disp.end();
				dg_cell = history_dof_handler.begin_active();
				for (; cell_me != endc_me; ++cell_me, ++dg_cell)
					if (cell_me->is_locally_owned())
					{
						unsigned int dofs_per_cell_disp = cell_me->get_fe().dofs_per_cell;
						if (dofs_per_cell_disp != 0)
						{
						PointHistory<dim> *local_quadrature_points_history
															= reinterpret_cast<PointHistory<dim> *>(cell_me->user_pointer());
						Assert (local_quadrature_points_history >=
									&quadrature_point_history.front(), ExcInternalError());
						Assert (local_quadrature_points_history <
									&quadrature_point_history.back(), ExcInternalError());
						for (unsigned int i=0; i<dim; i++)
							for (unsigned int j=0; j<dim; j++)
							{
								dg_cell->get_dof_values (history_stress_field[i][j],
														local_history_stress_fe_values[i][j]);
								dof_to_qpoint_matrix.vmult (local_history_stress_values_at_qpoints[i][j],
														local_history_stress_fe_values[i][j]);

								dg_cell->get_dof_values (history_strain_field[i][j],
														local_history_strain_fe_values[i][j]);
								dof_to_qpoint_matrix.vmult (local_history_strain_values_at_qpoints[i][j],
														local_history_strain_fe_values[i][j]);
								for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
								{
									local_quadrature_points_history[q].pre_stress[i][j]
																					 = local_history_stress_values_at_qpoints[i][j](q);

									local_quadrature_points_history[q].pre_plastic_strain[i][j]
																					 = local_history_strain_values_at_qpoints[i][j](q);
									local_quadrature_points_history[q].old_stress[i][j]
																					 = local_history_stress_values_at_qpoints[i][j](q);

									local_quadrature_points_history[q].old_plastic_strain[i][j]
																					 = local_history_strain_values_at_qpoints[i][j](q);
								}
							}

						dg_cell->get_dof_values (history_epstrain_field,
												local_history_epstrain_fe_values);
						dof_to_qpoint_matrix.vmult (local_history_epstrain_values_at_qpoints,
												local_history_epstrain_fe_values);
//						cout<<"local_history_epstrain_fe_values ="<<local_history_epstrain_fe_values.l2_norm()<<endl;
						for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
						{
							local_quadrature_points_history[q].pre_effective_plastic_strain
																			 = local_history_epstrain_values_at_qpoints(q);
							local_quadrature_points_history[q].old_effective_plastic_strain
																			 = local_history_epstrain_values_at_qpoints(q);
//							cout<<"local_history_epstrain(q)="<<local_history_epstrain_values_at_qpoints(q)<<endl;
						}
//						cout<<"local_history_epstrain ="<<local_history_epstrain_values_at_qpoints.l2_norm()<<endl;
						}
					}

//	  }

//				if (timestep_number > 1){
//					timestep_number +=1000;
//					timestep_number += step;
//					move_mesh(true);	// distortion
//					output_mech_results();
//					move_mesh(false);	// go back to the original mesh
//					timestep_number -= 1000;
//					timestep_number -= step;
//				}
	  }

  }

  template<int dim>
  void HeatEquation<dim>::refine_mesh_btw_layers(const double part_height_before_activate_next_layer, const int max_refine_level)
  {// only refine, no coarsening
	  for (int step=0; step<max_refine_level; ++step)
	  {
		  unsigned int cnt_refined_cells(0);
  		  for (auto cell: triangulation.active_cell_iterators())
  		  {
  			  if (cell->level() >= max_refine_level)
  				  continue;

//  			  Point<dim> centerPt = cell->center();
//  			  if (centerPt[2] > part_height_before_activate_next_layer)
  				  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
  				  {
  					  const Point<dim> face_center = cell->face(f)->center();

  					  if (fabs(face_center[2] - part_height_before_activate_next_layer) < 1e-6 )//&& 	// z=0
//  						  fabs(face_center[dim -2]) < EquationData::g_width/2. &&	// |y|<g_width
//						  fabs(face_center[dim -3]) < EquationData::g_length/2.)		// |x|<g_length
  					  {
  						  cell->set_refine_flag ();
  						  cnt_refined_cells++;
  						  break;
  					  }
  				  }
  		  }
  		  if(cnt_refined_cells == 0) // if no cell is going to be refined on this level, then continue to the next level
  			  continue;

  		  //--- for mechanical analysis ---history variables on quadrature points-start
//  		  if(thermal_mechanical_flg)
//  		 {
  			  // ---------------------------------------------------------------
  			  // Make a field variable for history varibales to be able to transfer the data to the quadrature points of the new mesh
			  FE_DGQ<dim> history_fe (1);
			  DoFHandler<dim> history_dof_handler (triangulation);
			  history_dof_handler.distribute_dofs (history_fe);

			  std::vector< std::vector< Vector<double> > >
								history_stress_field (dim, std::vector< Vector<double> >(dim)),		// pre_stress
								local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
								local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim));
			  std::vector< std::vector< Vector<double> > >
								history_strain_field (dim, std::vector< Vector<double> >(dim)),		// pre_plastic_strain
								local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
								local_history_strain_fe_values (dim, std::vector< Vector<double> >(dim));
			  Vector< double>
								history_epstrain_field (history_dof_handler.n_dofs()),		// effective_plastic_strain (double)
								local_history_epstrain_values_at_qpoints (quadrature_collection_disp[0].size()),
								local_history_epstrain_fe_values (history_fe.dofs_per_cell);

			  IndexSet history_locally_owned_dofs = history_dof_handler.locally_owned_dofs ();
			  LA::MPI::Vector mpi_history_epstrain_field(history_locally_owned_dofs, mpi_communicator);

			  std::vector<LA::MPI::Vector> mpi_history_field_row(dim);
			  std::vector< std::vector<LA::MPI::Vector> > mpi_history_stress_field(dim, mpi_history_field_row);//std::vector< Vector<double> >(dim, mpi_history_epstrain_field));
			  std::vector< std::vector<LA::MPI::Vector> > mpi_history_strain_field(dim, mpi_history_field_row);//(dim);//(dim, std::vector< Vector<double> >(dim));

			  for (unsigned int i=0; i<dim; i++)
				  for (unsigned int j=0; j<dim; j++)
				  {
					  mpi_history_stress_field[i][j].reinit(history_locally_owned_dofs, mpi_communicator);
					  mpi_history_strain_field[i][j].reinit(history_locally_owned_dofs, mpi_communicator);
				  }

			  for (unsigned int i=0; i<dim; i++)
				for (unsigned int j=0; j<dim; j++)
				{
					history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
					local_history_stress_values_at_qpoints[i][j].reinit(quadrature_collection_disp[0].size());
					local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);

					history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
					local_history_strain_values_at_qpoints[i][j].reinit(quadrature_collection_disp[0].size());
					local_history_strain_fe_values[i][j].reinit(history_fe.dofs_per_cell);
				}

			 FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
																quadrature_collection_disp[0].size());
			 FETools::compute_projection_from_quadrature_points_matrix(history_fe,
								quadrature_collection_disp[0], quadrature_collection_disp[0], qpoint_to_dof_matrix);

			 typename hp::DoFHandler<dim>::active_cell_iterator
											cell_me = dof_handler_disp.begin_active(),
											endc_me = dof_handler_disp.end();
			 typename DoFHandler<dim>::active_cell_iterator
											dg_cell = history_dof_handler.begin_active();
			 for (; cell_me!=endc_me; ++cell_me, ++dg_cell)
				if (cell_me->is_locally_owned())
				{
					const unsigned int dofs_per_cell_disp = cell_me -> get_fe().dofs_per_cell;
					if (dofs_per_cell_disp != 0) // To skip the cell with FE = FE_Nothing because there is no support point there
					{
					PointHistory<dim> *local_quadrature_points_history
														= reinterpret_cast<PointHistory<dim> *>(cell_me->user_pointer());
					Assert (local_quadrature_points_history >=
						   &quadrature_point_history.front(), ExcInternalError());
					Assert (local_quadrature_points_history <
						   &quadrature_point_history.back(), ExcInternalError());
					for (unsigned int i=0; i<dim; i++)
						for (unsigned int j=0; j<dim; j++)
						{
							for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
							{
								local_history_stress_values_at_qpoints[i][j](q)
											  = local_quadrature_points_history[q].pre_stress[i][j];

								local_history_strain_values_at_qpoints[i][j](q)
											  = local_quadrature_points_history[q].pre_plastic_strain[i][j];
							}
							qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
													 local_history_stress_values_at_qpoints[i][j]);
							dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
													history_stress_field[i][j]);
							qpoint_to_dof_matrix.vmult (local_history_strain_fe_values[i][j],
													 local_history_strain_values_at_qpoints[i][j]);
							dg_cell->set_dof_values (local_history_strain_fe_values[i][j],
													history_strain_field[i][j]);
						}
					// mapping effective plastic strain from quadrature points to Dof
					for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
					{
						local_history_epstrain_values_at_qpoints(q)
											  = local_quadrature_points_history[q].pre_effective_plastic_strain;
					}
					qpoint_to_dof_matrix.vmult (local_history_epstrain_fe_values,
												local_history_epstrain_values_at_qpoints);
					dg_cell->set_dof_values (local_history_epstrain_fe_values,
												history_epstrain_field);
					}
				}
			 // added-0606-start--update local to global
			 for (unsigned int i=0; i<dim; i++)
				 for (unsigned int j=0; j<dim; j++)
				 {
					 //stress
					 mpi_history_stress_field[i][j] = history_stress_field[i][j];
					 history_stress_field[i][j] = mpi_history_stress_field[i][j];
					 //strain
					 mpi_history_strain_field[i][j] = history_strain_field[i][j];
					 history_strain_field[i][j] = mpi_history_strain_field[i][j];
				 }
			 // effective plastic strain
			 mpi_history_epstrain_field = history_epstrain_field;
			 history_epstrain_field = mpi_history_epstrain_field;
			 // added-0606-end--update local to global
//  		 }
  		 //--- for mechanical analysis ---history variables on quadrature points-end

  		  triangulation.prepare_coarsening_and_refinement();

  		  // material id transfer for material_id --- start
  		  SolutionTransfer<dim, Vector<double>, DoFHandler<dim>>
//  	    SolutionTransfer<dim, LA::MPI::Vector, DoFHandler<dim>>
  							material_field_transfer(material_dof_handler);
  	    Vector< double> material_field (material_dof_handler.n_dofs());

  	    unsigned int dofs_per_cell = history_fe_material.dofs_per_cell;

  		LA::MPI::Vector distributed_material_id_per_cell;

  		std::vector<bool> point_saved(material_dof_handler.n_dofs(), false);
  		distributed_material_id_per_cell.reinit(material_locally_owned_dofs, //material_locally_relevant_dofs,
  																		mpi_communicator);
  		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  		unsigned int cnt_cells_for_material(0);//, active_cnt_cells(0);
  		for (typename DoFHandler<dim>::active_cell_iterator
  	   				cell = material_dof_handler.begin_active();
  	   				cell!=material_dof_handler.end(); ++cell, ++cnt_cells_for_material)
  		{
  			if (cell->is_locally_owned())
  			{
  				cell->get_dof_indices (local_dof_indices);
  				for (unsigned int i=0; i<dofs_per_cell; ++i)
  				{
  					if (point_saved[local_dof_indices[i]] == true)
  						continue;
  					distributed_material_id_per_cell(local_dof_indices[i]) = cell_material[cnt_cells_for_material];
  					point_saved[local_dof_indices[i]] = true;
  				}
  			}
  		}
  		distributed_material_id_per_cell.compress (VectorOperation::insert);
  		material_field = distributed_material_id_per_cell;

//  		LA::MPI::Vector local_solution_material(material_locally_owned_dofs, material_locally_relevant_dofs,
//  		            											mpi_communicator);
//  		LA::MPI::Vector local_solution_material_no_ghost_vector(material_locally_owned_dofs, //locally_relevant_dofs,
//  		            											mpi_communicator);
//  		local_solution_material_no_ghost_vector = material_field;
//  		local_solution_material = local_solution_material_no_ghost_vector;
//  		material_field_transfer.prepare_for_coarsening_and_refinement(local_solution_material);
  		material_field_transfer.prepare_for_coarsening_and_refinement(material_field);
  	     // material id transfer for material_id --- end

  	     // solution transfer for heat (dof_handler)
  	     SolutionTransfer<dim, Vector<double>, hp::DoFHandler<dim>>
  				solution_trans(dof_handler);
  	    solution_trans.prepare_for_coarsening_and_refinement(solution);//(previous_solution);
//  	  cout<<"refine btw layers - test-04"<<endl;
  	     //--- for mechanical analysis ---solutionTransfer-start
//  	    if(thermal_mechanical_flg)
//  	    {
			 SolutionTransfer<dim, Vector<double>, hp::DoFHandler<dim>>
					solution_trans_disp(dof_handler_disp);

			 solution_trans_disp.prepare_for_coarsening_and_refinement(solution_disp);

			 SolutionTransfer<dim, Vector<double> >
							history_stress_field_transfer0(history_dof_handler),
							history_stress_field_transfer1(history_dof_handler),
							history_stress_field_transfer2(history_dof_handler);
			 history_stress_field_transfer0.prepare_for_coarsening_and_refinement(history_stress_field[0]);
			 if ( dim > 1)
			 {
				history_stress_field_transfer1.prepare_for_coarsening_and_refinement(history_stress_field[1]);
			 }
			 if ( dim == 3)
			 {
				history_stress_field_transfer2.prepare_for_coarsening_and_refinement(history_stress_field[2]);
			 }

			 SolutionTransfer<dim, Vector<double> >
							history_strain_field_transfer0(history_dof_handler),
							history_strain_field_transfer1(history_dof_handler),
							history_strain_field_transfer2(history_dof_handler);
			 history_strain_field_transfer0.prepare_for_coarsening_and_refinement(history_strain_field[0]);
			 if ( dim > 1)
			 {
				history_strain_field_transfer1.prepare_for_coarsening_and_refinement(history_strain_field[1]);
			 }
			 if ( dim == 3)
			 {
				history_strain_field_transfer2.prepare_for_coarsening_and_refinement(history_strain_field[2]);
			 }

			 SolutionTransfer<dim, Vector<double> >
							history_epstrain_field_transfer(history_dof_handler);
			 history_epstrain_field_transfer.prepare_for_coarsening_and_refinement(history_epstrain_field);
//  	    }
  	     //--- for mechanical analysis ---solutionTransfer-end

  	    triangulation.execute_coarsening_and_refinement ();

//  	    if(thermal_mechanical_flg)
  	    	setup_quadrature_point_history (); // for mechanical analysis

  	    // added in 2019-0207 --start
  	    set_active_fe_indices();
  	    setup_system();
  	    // added in 2019-0207 --end
  	    // added in 2019-0527 --start
//  	    if(thermal_mechanical_flg)
  	    	Vector<double> tmp_solution_disp = solution_disp;
  	    	setup_mech_system();
  	    // added in 2019-0527 --end

  	    /////////---------material id transfer --- start
//  	    material_dof_handler.distribute_dofs (history_fe_material);
//
//   	    material_locally_owned_dofs = material_dof_handler.locally_owned_dofs ();
//   	    DoFTools::extract_locally_relevant_dofs (material_dof_handler,
//   													 material_locally_relevant_dofs);
//   		std::vector<LA::MPI::Vector> vec_solution_material(1), vec_new_solution_material(1);
//   		vec_solution_material[0] = local_solution_material;
//
//   		vec_new_solution_material[0].reinit (material_locally_owned_dofs,//, locally_relevant_dofs,
//   	                       mpi_communicator);
//   		material_field_transfer.interpolate(vec_solution_material, vec_new_solution_material);
//
//   	    vec_new_solution_material[0].compress (VectorOperation::insert);
//   	    material_field.reinit(material_dof_handler.n_dofs());
//   	    material_field = vec_new_solution_material[0];

  	    Vector<double> new_material_field(material_dof_handler.n_dofs());
  	    material_field_transfer.interpolate(material_field, new_material_field);

  	    material_field = new_material_field;

//   	 cout<<"refine btw layers - test-05"<<endl;

//   	    cell_material.reinit(triangulation.n_active_cells());
   		cnt_cells_for_material = 0;//, active_cnt_cells(0);
   		for (typename DoFHandler<dim>::active_cell_iterator
   	   				cell = material_dof_handler.begin_active();
   	   				cell!=material_dof_handler.end(); ++cell, ++cnt_cells_for_material)
   		{
   			if (cell->is_locally_owned())
   			{
   				cell->get_dof_indices (local_dof_indices);
   				unsigned int powder_count = 0, solid_count = 0, liquid_count = 0;
   				for (unsigned int i=0; i<dofs_per_cell; ++i)
   				{
   					if(material_field(local_dof_indices[i]) == 2)
   						powder_count++;
   					else if(material_field(local_dof_indices[i]) == 1)
   						liquid_count++;
   					else if(material_field(local_dof_indices[i]) == 0)
   						solid_count++;
   				}
   				unsigned int material_type = 0;
   				if(powder_count > liquid_count && powder_count > solid_count)
   					material_type = 2;
   				else if(solid_count > liquid_count && solid_count > powder_count)
   					material_type = 0;
   				else if(liquid_count > powder_count && liquid_count > solid_count)
   					material_type = 1;
   				cell_material[cnt_cells_for_material] = material_type;
   	//			distributed_material_id_per_cell[cnt_cells_for_material] = material_type;
   				cell->set_material_id(material_type);
   			}
   		}
   		/////////---------material id transfer --- end

//   		cout<<"refine btw layers - test-06"<<endl;
//  	    set_active_fe_indices();

//  	    dof_handler.distribute_dofs(fe_collection);
//  	    locally_owned_dofs = dof_handler.locally_owned_dofs ();
//  	    DoFTools::extract_locally_relevant_dofs (dof_handler,
//  	    										locally_relevant_dofs);
//  	    dof_handler_disp.distribute_dofs (fe_collection_disp);


  	    // Vector to visualize the material type of each cell
//  	    cell_material.reinit(triangulation.n_active_cells());
  	    // Vector to visualize the FE of each cell
//  	    FE_Type.reinit(triangulation.n_active_cells());
//  	    unsigned int cnt_cells = 0;
//  	    typename hp::DoFHandler<dim>::active_cell_iterator
//  		  	  cell = dof_handler.begin_active(),
//  			  endc = dof_handler.end();
//  	    for (; cell!=endc; ++cell)
//  	    {
////  	    	unsigned int material_id = cell->material_id();
////  	    	cell_material[cnt_cells] = material_id;
//
//  	    	unsigned int fe_index = cell->active_fe_index();
//  	    	FE_Type[cnt_cells] = fe_index;
//  	    	++ cnt_cells;
//  	    }


   	    pcout <<"cell number after refinement: "<< cnt_cells_for_material<<endl;
//   	 ++timestep_number;output_results();return;

  	    // Heat Solution interpolation on the new Dof handler --- start
  	    Vector<double> new_solution(dof_handler.n_dofs());
  	    solution_trans.interpolate(solution, new_solution);
  	    solution.reinit(dof_handler.n_dofs());
  	    solution = new_solution;

//  	    constraints.clear();
//  	    constraints.reinit (locally_relevant_dofs);
//  	    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
//  	    VectorTools::interpolate_boundary_values(dof_handler,
//  	    												21,
//														EquationData::BoundaryValues<dim>(),
//														constraints); // interpolate Dirichlet boundary condition
//
//  	    constraints.close();
  	    // Computation of the hanging nodes constraints
  	    constraints.distribute (solution);
  	  // Heat Solution interpolation on the new Dof handler --- end

  	    // Displacement solution interpolation on the new Dofhandler_disp -- mechanical analysis
//  	    if(thermal_mechanical_flg)
//  	    {
			 Vector<double> distributed_solution_disp(dof_handler_disp.n_dofs());
			 solution_trans_disp.interpolate(tmp_solution_disp/*solution_disp*/, distributed_solution_disp);
			 solution_disp.reinit(dof_handler_disp.n_dofs());
			 solution_disp = distributed_solution_disp;

//			 constraints_disp.clear();
//			 DoFTools::make_hanging_node_constraints(dof_handler_disp, constraints_disp);
//			 constraints_disp.close();
//			 compute_dirichlet_constraints();
			 // Computation of the hanging nodes constraints
			 constraints_dirichlet_and_hanging_nodes.distribute (solution_disp);
			 // ---------------------------------------------------
			 history_dof_handler.distribute_dofs (history_fe);
			 // stress
			 std::vector< std::vector< Vector<double> > >
							distributed_history_stress_field(dim, std::vector< Vector<double> >(dim));
			 for (unsigned int i=0; i<dim; i++)
				for (unsigned int j=0; j<dim; j++)
				{
					distributed_history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
				}

			 history_stress_field_transfer0.interpolate(history_stress_field[0], distributed_history_stress_field[0]);
			 if ( dim > 1)
			 {
				history_stress_field_transfer1.interpolate(history_stress_field[1], distributed_history_stress_field[1]);
			 }
			 if ( dim == 3)
			 {
				history_stress_field_transfer2.interpolate(history_stress_field[2], distributed_history_stress_field[2]);
			 }

			 history_stress_field = distributed_history_stress_field;

			 // plastic strain
			 std::vector< std::vector< Vector<double> > >
						distributed_history_strain_field (dim, std::vector< Vector<double> >(dim));
			 for (unsigned int i=0; i<dim; i++)
				for (unsigned int j=0; j<dim; j++)
				{
					distributed_history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
				}

			 history_strain_field_transfer0.interpolate(history_strain_field[0], distributed_history_strain_field[0]);
			 if ( dim > 1)
			 {
				history_strain_field_transfer1.interpolate(history_strain_field[1], distributed_history_strain_field[1]);
			 }
			 if ( dim == 3)
			 {
				history_strain_field_transfer2.interpolate(history_strain_field[2], distributed_history_strain_field[2]);
			 }

			 history_strain_field = distributed_history_strain_field;

			 // effective plastic strain
			 Vector<double>
						distributed_history_epstrain_field (history_dof_handler.n_dofs());
			 history_epstrain_field_transfer.interpolate(history_epstrain_field, distributed_history_epstrain_field);

			 history_epstrain_field = distributed_history_epstrain_field;

			 // ---------------------------------------------------------------
			 // Transfer the history data to the quadrature points of the new mesh
			 // In a final step, we have to get the data back from the now
			 // interpolated global field to the quadrature points on the
			 // new mesh. The following code will do that:

			 FullMatrix<double> dof_to_qpoint_matrix (quadrature_collection_disp[0].size(),
													  history_fe.dofs_per_cell);
			 FETools::compute_interpolation_to_quadrature_points_matrix (history_fe,
													quadrature_collection_disp[0], dof_to_qpoint_matrix);

			 cell_me = dof_handler_disp.begin_active();
			 endc_me = dof_handler_disp.end();
			 dg_cell = history_dof_handler.begin_active();
			 for (; cell_me != endc_me; ++cell_me, ++dg_cell)
				if (cell_me->is_locally_owned())
				{
					unsigned int dofs_per_cell_disp = cell_me->get_fe().dofs_per_cell;
					if (dofs_per_cell_disp != 0)
					{
					PointHistory<dim> *local_quadrature_points_history
														= reinterpret_cast<PointHistory<dim> *>(cell_me->user_pointer());
					Assert (local_quadrature_points_history >=
								&quadrature_point_history.front(), ExcInternalError());
					Assert (local_quadrature_points_history <
								&quadrature_point_history.back(), ExcInternalError());
					for (unsigned int i=0; i<dim; i++)
						for (unsigned int j=0; j<dim; j++)
						{
							dg_cell->get_dof_values (history_stress_field[i][j],
													local_history_stress_fe_values[i][j]);
							dof_to_qpoint_matrix.vmult (local_history_stress_values_at_qpoints[i][j],
													local_history_stress_fe_values[i][j]);

							dg_cell->get_dof_values (history_strain_field[i][j],
													local_history_strain_fe_values[i][j]);
							dof_to_qpoint_matrix.vmult (local_history_strain_values_at_qpoints[i][j],
													local_history_strain_fe_values[i][j]);
							for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
							{
								local_quadrature_points_history[q].pre_stress[i][j]
																				 = local_history_stress_values_at_qpoints[i][j](q);

								local_quadrature_points_history[q].pre_plastic_strain[i][j]
																				 = local_history_strain_values_at_qpoints[i][j](q);
								local_quadrature_points_history[q].old_stress[i][j]
																				 = local_history_stress_values_at_qpoints[i][j](q);

								local_quadrature_points_history[q].old_plastic_strain[i][j]
																				 = local_history_strain_values_at_qpoints[i][j](q);
							}
						}

					dg_cell->get_dof_values (history_epstrain_field,
											local_history_epstrain_fe_values);
					dof_to_qpoint_matrix.vmult (local_history_epstrain_values_at_qpoints,
											local_history_epstrain_fe_values);
					for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
					{
						local_quadrature_points_history[q].pre_effective_plastic_strain
																		 = local_history_epstrain_values_at_qpoints(q);
						local_quadrature_points_history[q].old_effective_plastic_strain
																		 = local_history_epstrain_values_at_qpoints(q);
					}
					}
				}
//  	    }
	  }
  }

  template <int dim>
  void HeatEquation<dim>::store_old_vectors()
  {
	  TimerOutput::Scope timer_section(computing_timer, "store_old_vectors");

	  map_old_solution.clear();
//	  map_old_solution_disp.clear();
	  const MappingQ1<dim,dim> mapping;
	  std::vector<bool> point_saved(dof_handler.n_dofs(), false);
	  const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell;
	  // Vector of the support points of one cell
	  std::vector<Point<dim>> support_points(dofs_per_active_cell);
	  // Get the coordinates of the support points on the unit cell
	  support_points = fe_collection[0].get_unit_support_points();
	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_active_cell);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  cell1 = dof_handler.begin_active(),
		  endc1 = dof_handler.end();
	  for ( ;cell1!=endc1; cell1++)
	  {
		  // Temporary variable to get the number of dof for the currently visited cell
		  const unsigned int dofs_per_cell = cell1 -> get_fe().dofs_per_cell;
		  if (dofs_per_cell != 0) // To skip the cell with FE = FE_Nothing because there is no support point there
		  {
//			  std::vector<Point<dim>> support_points(dofs_per_cell);
//			  support_points = fe_collection[0].get_unit_support_points();
//			  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
			  cell1->get_dof_indices (local_dof_indices);
			  // temperary use to store the old solution on current cell
//			  Vector<double> temp_solution(dofs_per_cell);
			  for (unsigned int i=0;i<dofs_per_cell;i++)
			  {
 				  if (point_saved[local_dof_indices[i]] == true)
 					  continue;

				  //Get the coordinates of the support points on the real cell
				  Point<dim> support_point_i = mapping.transform_unit_to_real_cell(cell1, support_points[i]);
				  // Get the old solution on current point respect to the global index
				  map_old_solution[support_point_i] = old_solution(local_dof_indices[i]);	// store the old solution with respect to the coordinates
//				  Vector<double> old_solution_heat_disp(1 + dim);	//store node values: temperature + (u,v,w)
//				  old_solution_heat_disp[0] = old_solution(local_dof_indices[i]);
//				  old_solution_heat_disp[1] = solution_disp(local_dof_indices[i]);
//				  old_solution_heat_disp[2] = solution_disp(local_dof_indices[i]+1);
//				  old_solution_heat_disp[3] = solution_disp(local_dof_indices[i]+2);
//				  map_old_solution_disp[support_point_i] = old_solution_heat_disp;
				  point_saved[local_dof_indices[i]] = true;
			  }
		  }
	  }
//	  cout<<"map_old_solution.size() = "<<map_old_solution.size() <<std::endl;;
  }


  template <int dim>
   void HeatEquation<dim>::transfer_old_vectors()
   {
	  TimerOutput::Scope timer_section(computing_timer, "transfer_old_vectors");

 	  // Creation of a solution of the same size of the number of dof of the new FE space
 	  Vector<double> long_solution_heat;//, long_solution_disp;
 	  long_solution_heat.reinit(dof_handler.n_dofs(), false);
// 	  	  	  	  dof_handler_disp.distribute_dofs (fe_collection_disp);
// 	  long_solution_disp.reinit(dof_handler_disp.n_dofs(), false);
 	  long_solution_heat = Tinit;	// set the new activated cells to inital temperature

 	  std::vector<bool> point_visited(dof_handler.n_dofs(), false);

 	 const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell;
 	  // Vector of the support points of one cell
 	  std::vector<Point<dim>> support_points(dofs_per_active_cell);
 	 // Get the coordinates of the support points on the unit cell
 	  support_points = fe_collection[0].get_unit_support_points();
 	  // Vector of the degree of freedom indices of one cell
 	 std::vector<types::global_dof_index> local_dof_indices;
 	  local_dof_indices.resize(dofs_per_active_cell);
// 	 cout<<"transfer test 0 !!!!!"<<endl;
 	  double current_max_height = part_height + thickness/2.0;

 	  const MappingQ1<dim,dim> mapping;
 	  typename hp::DoFHandler<dim>::active_cell_iterator
 	  	  cell = dof_handler.begin_active(),
 		  endc = dof_handler.end();
 	  for ( ;cell!=endc; cell++)
 	  {
 		  // Number of dof of the currently visited cell
 		  const unsigned int dofs_per_cell = cell ->get_fe().dofs_per_cell;
// 		  // Vector of the support points of one cell
// 		  std::vector<Point<dim>> support_points(dofs_per_cell);
 		  if (dofs_per_cell != 0) // To skip the cell with FE = FE_Nothing because they have not any support point
 		  {
// 			  // Get the coordinates of the support points on the unit cell
// 			  support_points = fe_collection[0].get_unit_support_points();
// 			  // Vector of the degree of freedom indices of one cell
// 			  local_dof_indices.resize(dofs_per_cell);
 			  cell->get_dof_indices(local_dof_indices);

 			  //Get the coordinates of the support points on the real cell
 			  for (unsigned int i=0;i<dofs_per_cell;i++)
 			  {
 				  if (point_visited[local_dof_indices[i]] == true)
 					  continue;

 				  Point<dim> support_points_i = mapping.transform_unit_to_real_cell(cell, support_points[i]);
 				  if (support_points_i[2] > current_max_height)
 				  {
// 					 long_solution[local_dof_indices[i]]  = Tinit;
 					 point_visited[local_dof_indices[i]] = true;
 					  continue;
 				  }

// 				  typename std::map< Point<dim>, double>::iterator
// 				  	  iter_old = map_old_solution.begin();
// 				  typename std::map< Point<dim>, Vector<double>>::iterator
//				  	  iterator_old_heat_me = map_old_solution_disp.begin();


 				    auto search = map_old_solution.find(support_points_i);
 				    if (search != map_old_solution.end())
 				    {
// 				        std::cout << "Found " << search->first << " " << search->second << '\n';
 				    	long_solution_heat[local_dof_indices[i]] = search->second; //map_old_solution.find(support_points_i)->second;  //
// 				    	long_solution_heat[local_dof_indices[i]] = iterator_old_heat_me->second[0];
// 				    	long_solution_disp[local_dof_indices[i]] = iterator_old_heat_me->second[1]; // u
// 				    	long_solution_disp[local_dof_indices[i]+1] = iterator_old_heat_me->second[2]; // v
// 				    	long_solution_disp[local_dof_indices[i]+2] = iterator_old_heat_me->second[3]; // w
 				    	point_visited[local_dof_indices[i]] = true;
// 				    	map_old_solution.erase(search);
// 				    	map_old_solution_disp.erase(iterator_old_heat_me);
 				    }
 				    else
 				    {
// 				    	std::cout << "Not found\n";
 				    }



// 				  // Iteration in the old solution map
// 				  for ( ; iter_old!= map_old_solution.end(); iter_old++, iterator_old_heat_me++)
// 				  {
// 					  // Test if the point visited corresponds to a point in the "old" dof_handler
// 					  if (support_points_i == iter_old->first)
// 					  {
// 						  // Write the solution at the right place inside vector
// 						  long_solution_heat[local_dof_indices[i]] = iter_old->second; //map_old_solution.find(support_points_i)->second;  //
// 						  long_solution_heat[local_dof_indices[i]] = iterator_old_heat_me->second[0];
// 						  long_solution_disp[local_dof_indices[i]] = iterator_old_heat_me->second[1]; // u
// 	 					  long_solution_disp[local_dof_indices[i]+1] = iterator_old_heat_me->second[2]; // v
// 	 					  long_solution_disp[local_dof_indices[i]+2] = iterator_old_heat_me->second[3]; // w
// 	 					  point_visited[local_dof_indices[i]] = true;
// 	 					  map_old_solution.erase(iter_old);
// 	 					  map_old_solution_disp.erase(iterator_old_heat_me);
//
// 						  break;
// 					  }
// 				  }
 			  }
 		  }
 	  }
// 	 cout<<"transfer test 1 !!!!!"<<endl;
 	  solution.reinit(dof_handler.n_dofs());
 	  old_solution = long_solution_heat;
 	  constraints.clear();
 	  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
 	  constraints.close();
 	  constraints.distribute (old_solution);
// 	 cout<<"transfer test 2 !!!!!"<<endl;
// 	  if(thermal_mechanical_flg)
// 	  {
// 		  solution_disp.reinit(dof_handler_disp.n_dofs());
// 		  solution_disp = long_solution_disp;
//		  constraints_disp.clear();
//		  DoFTools::make_hanging_node_constraints(dof_handler_disp, constraints_disp);
//		 cout<<"transfer test 3 !!!!!"<<endl;
//		  constraints_disp.close();
//		 compute_dirichlet_constraints();
//	// 	cout<<"transfer test 4 !!!!!"<<endl;
//		 constraints_dirichlet_and_hanging_nodes.distribute (solution_disp);
// 	  }
   }



  template <int dim>
  void HeatEquation<dim>::store_old_vectors_disp()
  {
	  TimerOutput::Scope timer_section(computing_timer, "store_old_vectors_disp");

	  map_old_solution_disp.clear();
	  const MappingQ1<dim,dim> mapping;
	  std::vector<bool> point_saved_disp(dof_handler_disp.n_dofs(), false);

	  const unsigned int 			dofs_per_active_cell_disp = fe_collection_disp[0].dofs_per_cell;
	  // Vector of the support points of one cell
	  std::vector<Point<dim>> support_points_disp(dofs_per_active_cell_disp);
	  // Get the coordinates of the support points on the unit cell
	  support_points_disp = fe_collection_disp[0].get_unit_support_points();

//	  pcout<<"dof_handler_disp.n_dofs():"<<dof_handler_disp.n_dofs()<<endl;
//	  pcout<<"dofs_per_active_cell_disp:"<<dofs_per_active_cell_disp<<endl;
//	  for(unsigned int t=0; t< support_points_disp.size(); t++)
//		  pcout<<"support_points_disp:"<<support_points_disp[t] <<", ";
//	  pcout<<endl;
	  unsigned int cnt_active = 0;

	  std::vector<types::global_dof_index> local_dof_indices_disp (dofs_per_active_cell_disp);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  cell1 = dof_handler_disp.begin_active(),
		  endc1 = dof_handler_disp.end();
	  for ( ;cell1!=endc1; cell1++)
	  {
		  if (cell1->is_locally_owned())
		  {
		  // Temporary variable to get the number of dof for the currently visited cell
		  const unsigned int dofs_per_cell = cell1 -> get_fe().dofs_per_cell;
		  if (dofs_per_cell != 0) // To skip the cell with FE = FE_Nothing because there is no support point there
		  {
			  cnt_active++;
//			  std::vector<Point<dim>> support_points(dofs_per_cell);
//			  support_points = fe_collection[0].get_unit_support_points();
//			  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
			  cell1->get_dof_indices (local_dof_indices_disp);
			  // temperary use to store the old solution on current cell
//			  Vector<double> temp_solution(dofs_per_cell);
			  for (unsigned int i=0;i<dofs_per_cell;i++)
			  {
 				  if (point_saved_disp[local_dof_indices_disp[i]] == true)
 				  {
 					  i += 2;
 					  continue;
 				  }

				  //Get the coordinates of the support points on the real cell
				  Point<dim> support_point_i = mapping.transform_unit_to_real_cell(cell1, support_points_disp[i]);
				  // Get the old solution on current point respect to the global index
//				  map_old_solution[support_point_i] = old_solution(local_dof_indices[i]);	// store the old solution with respect to the coordinates
				  Vector<double> old_solution_heat_disp(dim);	//store node values: temperature + (u,v,w)
//				  old_solution_heat_disp[0] = old_solution(local_dof_indices_disp[i]);
				  old_solution_heat_disp[0] = solution_disp(local_dof_indices_disp[i]);
				  old_solution_heat_disp[1] = solution_disp(local_dof_indices_disp[i]+1);
				  old_solution_heat_disp[2] = solution_disp(local_dof_indices_disp[i]+2);
				  map_old_solution_disp[support_point_i] = old_solution_heat_disp;
				  point_saved_disp[local_dof_indices_disp[i]] = true;
				  i += 2;
			  }
		  }

		  }
	  }
	  cout<<"cnt_active:"<<cnt_active<<", solution_disp.l2_norm() = "<<solution_disp.l2_norm()<<endl;
  }


  template <int dim>
   void HeatEquation<dim>::transfer_old_vectors_disp()
   {
	  TimerOutput::Scope timer_section(computing_timer, "transfer_old_vectors_disp");

 	  // Creation of a solution of the same size of the number of dof of the new FE space
 	  Vector<double> long_solution_disp;
// 	  long_solution_heat.reinit(dof_handler.n_dofs(), false);
// 	  	  	  	  dof_handler_disp.distribute_dofs (fe_collection_disp);
 	  long_solution_disp.reinit(dof_handler_disp.n_dofs(), false);
// 	  long_solution_heat = Tinit;	// set the new activated cells to inital temperature
 	  long_solution_disp = 0;

// 	  std::vector<bool> point_visited(dof_handler.n_dofs(), false);
 	  std::vector<bool> point_visited_disp(dof_handler_disp.n_dofs(), false);

// 	 const unsigned int 			dofs_per_active_cell = fe_collection[0].dofs_per_cell;
 	 const unsigned int 			dofs_per_active_cell_disp = fe_collection_disp[0].dofs_per_cell;
 	  // Vector of the support points of one cell
// 	  std::vector<Point<dim>> support_points(dofs_per_active_cell);
 	 std::vector<Point<dim>> support_points_disp(dofs_per_active_cell_disp);
 	 // Get the coordinates of the support points on the unit cell
// 	  support_points = fe_collection[0].get_unit_support_points();
 	 support_points_disp = fe_collection_disp[0].get_unit_support_points();
 	  // Vector of the degree of freedom indices of one cell
// 	 std::vector<types::global_dof_index> local_dof_indices;
 	std::vector<types::global_dof_index> local_dof_indices_disp;
// 	  local_dof_indices.resize(dofs_per_active_cell);
 	 local_dof_indices_disp.resize(dofs_per_active_cell_disp);
 	 pcout<<"transfer test 0 !!!!!"<<endl;
 	  double current_max_height = part_height + thickness/2.0;

 	 unsigned int cnt_active = 0;

 	  const MappingQ1<dim,dim> mapping;
 	  // for mechanical
 	  typename hp::DoFHandler<dim>::active_cell_iterator
 	  	  cell = dof_handler_disp.begin_active(),
 		  endc = dof_handler_disp.end();
 	  for ( ;cell!=endc; cell++)
 	  {
 		 if (cell->is_locally_owned())
 		 {
 		  // Number of dof of the currently visited cell
 		  const unsigned int dofs_per_cell = cell ->get_fe().dofs_per_cell;
// 		  // Vector of the support points of one cell
// 		  std::vector<Point<dim>> support_points(dofs_per_cell);
 		  if (dofs_per_cell != 0) // To skip the cell with FE = FE_Nothing because they have not any support point
 		  {
 			  cnt_active++;
// 			  // Get the coordinates of the support points on the unit cell
// 			  support_points = fe_collection[0].get_unit_support_points();
// 			  // Vector of the degree of freedom indices of one cell
// 			  local_dof_indices.resize(dofs_per_cell);
 			  cell->get_dof_indices(local_dof_indices_disp);

 			  //Get the coordinates of the support points on the real cell
 			  for (unsigned int i=0;i<dofs_per_cell;i++)
 			  {
 				  if (point_visited_disp[local_dof_indices_disp[i]] == true)
 				  {
 					  i += 2;
 					  continue;
 				  }

 				  Point<dim> support_points_i = mapping.transform_unit_to_real_cell(cell, support_points_disp[i]);
 				  if (support_points_i[2] > current_max_height)
 				  {
// 					 long_solution[local_dof_indices[i]]  = Tinit;
 					 point_visited_disp[local_dof_indices_disp[i]] = true;
 					 i += 2;
 					 continue;
 				  }

// 				  typename std::map< Point<dim>, double>::iterator
// 				  	  iter_old = map_old_solution.begin();
// 				  typename std::map< Point<dim>, Vector<double>>::iterator
//				  	  iterator_old_heat_me = map_old_solution_disp.begin();


 				    auto search = map_old_solution_disp.find(support_points_i);
 				    if (search != map_old_solution_disp.end())
 				    {
// 				        std::cout << "Found " << search->first << " " << search->second << '\n';
// 				    	long_solution_heat[local_dof_indices[i]] = search->second; //map_old_solution.find(support_points_i)->second;  //
 				    	long_solution_disp[local_dof_indices_disp[i]] = search->second[0];
 				    	long_solution_disp[local_dof_indices_disp[i] + 1] = search->second[1];
 				    	long_solution_disp[local_dof_indices_disp[i] + 2] = search->second[2];
// 				    	long_solution_disp[local_dof_indices[i]] = iterator_old_heat_me->second[1]; // u
// 				    	long_solution_disp[local_dof_indices[i]+1] = iterator_old_heat_me->second[2]; // v
// 				    	long_solution_disp[local_dof_indices[i]+2] = iterator_old_heat_me->second[3]; // w
 				    	point_visited_disp[local_dof_indices_disp[i]] = true;
 				    	i += 2;
// 				    	map_old_solution.erase(search);
// 				    	map_old_solution_disp.erase(iterator_old_heat_me);
 				    }
 				    else
 				    {
 				    	i += 2;
// 				    	std::cout << "Not found\n";
 				    }



// 				  // Iteration in the old solution map
// 				  for ( ; iter_old!= map_old_solution.end(); iter_old++, iterator_old_heat_me++)
// 				  {
// 					  // Test if the point visited corresponds to a point in the "old" dof_handler
// 					  if (support_points_i == iter_old->first)
// 					  {
// 						  // Write the solution at the right place inside vector
// 						  long_solution_heat[local_dof_indices[i]] = iter_old->second; //map_old_solution.find(support_points_i)->second;  //
// 						  long_solution_heat[local_dof_indices[i]] = iterator_old_heat_me->second[0];
// 						  long_solution_disp[local_dof_indices[i]] = iterator_old_heat_me->second[1]; // u
// 	 					  long_solution_disp[local_dof_indices[i]+1] = iterator_old_heat_me->second[2]; // v
// 	 					  long_solution_disp[local_dof_indices[i]+2] = iterator_old_heat_me->second[3]; // w
// 	 					  point_visited[local_dof_indices[i]] = true;
// 	 					  map_old_solution.erase(iter_old);
// 	 					  map_old_solution_disp.erase(iterator_old_heat_me);
//
// 						  break;
// 					  }
// 				  }
 			  }
 		  }

 		 }
 	  }

// 	 cout<<"cnt_active:"<<cnt_active<<endl;

// 	 pcout<<"transfer test 1 !!!!!"<<endl;
// 	  solution.reinit(dof_handler.n_dofs());
// 	  old_solution = long_solution_heat;
// 	  constraints.clear();
// 	  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
// 	  constraints.close();
// 	  constraints.distribute (old_solution);
// 	 pcout<<"transfer test 2 !!!!!"<<endl;
// 	  if(thermal_mechanical_flg)
// 	  {
 	 	LA::MPI::Vector mpi_long_solution_disp(locally_owned_dofs_disp, mpi_communicator);
		mpi_long_solution_disp = long_solution_disp;
		long_solution_disp = mpi_long_solution_disp;

		cout<<"cnt_active:"<<cnt_active<<", solution_disp.l2_norm() = "<<long_solution_disp.l2_norm()<<endl;

 		  solution_disp.reinit(dof_handler_disp.n_dofs());
 		  solution_disp = long_solution_disp;
//		  constraints_disp.clear();
//		  DoFTools::make_hanging_node_constraints(dof_handler_disp, constraints_disp);
//		 pcout<<"transfer test 3 !!!!!"<<endl;
//		  constraints_disp.close();
//		 compute_dirichlet_constraints();
//	 	pcout<<"transfer test 4 !!!!!"<<endl;
		 constraints_dirichlet_and_hanging_nodes.distribute (solution_disp);
// 	  }
   }




  template<int dim>
  void HeatEquation<dim>::get_attributes_in_layer_node(const xml_node<> *layer_node)
  {
	  layer_id = atoi(layer_node->first_node("Layer_id")->value());
	  total_scan_tracks = atof(layer_node->first_node("Total_scan_tracks")->value());
	  orientation = atof(layer_node->first_node("Orientation")->value());
	  thickness = atof(layer_node->first_node("Thickness")->value());
	  hatching_space = atof(layer_node->first_node("Hatching_space")->value());
	  part_height = atof(layer_node->first_node("Part_height")->value());
	  layer_start_time = atof(layer_node->first_node("Layer_start_time")->value());
	  layer_end_time = atof(layer_node->first_node("Layer_end_time")->value());
	  idle_time = atof(layer_node->first_node("Idle_time")->value());

	  std::string start_point_str = layer_node->first_node("Layer_start_point")->value();
	  std::string end_point_str = layer_node->first_node("Layer_end_point")->value();
	  std::vector<double> start_vec, end_vec;
	  double conversion_result;
	  std::istringstream is_start(start_point_str), is_end(end_point_str);
	  while(is_start >> conversion_result)
		  start_vec.push_back(conversion_result);
	  while(is_end>>conversion_result)
		  end_vec.push_back(conversion_result);
	  for (unsigned int coord = 0; coord < dim; coord++)
	  {
		  layer_start_point[coord] = start_vec[coord];
		  layer_end_point[coord] = end_vec[coord];
	  }

	  {
		  xml_node<> *scan_track_id = layer_node->next_sibling();
		  xml_node<> *track_node = scan_track_id->first_node();
		  xml_node<> *segment_id = track_node->next_sibling();
		  std::string start_point = segment_id->first_node("Segment_start_point")->value();
		  std::string end_point = segment_id->first_node("Segment_end_point")->value();
		  std::vector<double> start_vec, end_vec, print_direction_start(3);
		  double conversion_result;
		  std::istringstream is_start(start_point), is_end(end_point);
		  while(is_start >> conversion_result)
			  start_vec.push_back(conversion_result);
		  while(is_end>>conversion_result)
			  end_vec.push_back(conversion_result);
		  for (unsigned int coord = 0; coord < dim; coord++)
		  {
			  print_direction_start[coord] = (start_vec[coord] + end_vec[coord])/2.;
		  }
	//	  pcout<<"print_direction_start = " <<print_direction_start[0]<<", "<<print_direction_start[1]<<", "<<print_direction_start[2]<<std::endl;

		  xml_node<> *scan_track_2 = scan_track_id->next_sibling();
		  xml_node<> *segment_id2 = scan_track_2->first_node()->next_sibling();
		  start_point = segment_id2->first_node("Segment_start_point")->value();
		  end_point = segment_id2->first_node("Segment_end_point")->value();
		  std::vector<double> start_vec2, end_vec2, print_direction_end(3);
		  double conversion_result2;
		  std::istringstream is_start2(start_point), is_end2(end_point);
		  while(is_start2 >> conversion_result2)
			  start_vec2.push_back(conversion_result2);
		  while(is_end2>>conversion_result2)
			  end_vec2.push_back(conversion_result2);

		  Point<dim> start_Pt,  end_Pt,  toProject_Pt, Projection;
		  start_Pt[0] = start_vec2[0]; start_Pt[1] = start_vec2[1]; start_Pt[2] = start_vec2[2];
		  end_Pt[0] = end_vec2[0]; end_Pt[1] = end_vec2[1]; end_Pt[2] = end_vec2[2];
		  toProject_Pt[0] = print_direction_start[0]; toProject_Pt[1] = print_direction_start[1]; toProject_Pt[2] = print_direction_start[2];

		  Projection =  projection_point_to_line(start_Pt, end_Pt, toProject_Pt);

		  for (unsigned int coord = 0; coord < dim; coord++)
		  {
			  print_direction_end[coord] = Projection[coord];
		  }
	//	  pcout<<"print_direction_end = " <<print_direction_end[0]<<", "<<print_direction_end[1]<<", "<<print_direction_end[2]<<std::endl;

		  for (unsigned int coord = 0; coord < dim; coord++)
		  {
			  print_direction[coord] = print_direction_end[coord] - print_direction_start[coord];
		  }
		  pcout<<"print_direction = " <<print_direction<<", angle = "<<atan(print_direction[1]/print_direction[0])*180./M_PI<<", Orientation = "<<orientation*180./M_PI<<std::endl;  //acos(cosr)*180/PI;
	  }
  }

  template<int dim>
  void HeatEquation<dim>::get_attributes_in_track_node(const xml_node<> *track_node)
  {
		track_id = atoi(track_node->first_node("Track_id")->value());
		scan_velocity = atof(track_node->first_node("Scan_velocity")->value());
		time_step_point = atof(track_node->first_node("Time_step_point")->value());
		time_step_line = atof(track_node->first_node("Time_step_line")->value());
  }

  template<int dim>
  void HeatEquation<dim>::get_attributes_in_segment_node(const xml_node<> *segment_id)
  {
	  source_type = segment_id->first_node("Source_type")->value();
	  // if it is point heat source, then the "Segment_length" is the total length where point heat source will be applied,
	  // while if it is the line heat source, then the "Segment_length" is the length of each line heat input rather than the length where line heat source will be applied
	  segment_length = atof(segment_id->first_node("Segment_length")->value());
	  segment_start_time = atof(segment_id->first_node("Segment_start_time")->value());
	  segment_end_time = atof(segment_id->first_node("Segment_end_time")->value());
//	  cout<<"segment_start_time = "<<setprecision(20)<<segment_start_time<<endl;
//	  cout<<"segment_end_time = "<<setprecision(20)<<segment_end_time<<endl;
	   double  intpart;
	   double fractpart = modf (segment_start_time, &intpart);
	   fractpart  = roundf(fractpart * 1000000.0)/1000000.0;
	   segment_start_time = intpart + fractpart;

	   fractpart = modf (segment_end_time, &intpart);
	   fractpart  = roundf(fractpart * 1000000.0)/1000000.0;
	   segment_end_time = intpart + fractpart;
//		  cout<<"--segment_start_time = "<<setprecision(20)<<segment_start_time<<endl;
//		  cout<<"--segment_end_time = "<<setprecision(20)<<segment_end_time<<endl;

		if (source_type == "Point")
			time_step = time_step_point;
		else // source_type == "Line"
		{
			if (segment_length < min_segment_length)
			{
				source_type = "Point";
				time_step = time_step_point;
			}
			else if (segment_length > max_segment_length)
			{
				double temp_segment_length = segment_length;
				double n_div = 1;
				while (!(min_segment_length <= temp_segment_length && temp_segment_length <= max_segment_length))
				{
					n_div++;
					temp_segment_length = segment_length/n_div;
				}
				segment_length = temp_segment_length;
				cout << endl << "line segment length = " << segment_length <<endl;
			}
			time_step_line = segment_length/scan_velocity/num_dt_line;	// number of time steps for each line segment
//  					time_step = time_step_line;
		}

	  std::string start_point = segment_id->first_node("Segment_start_point")->value();
	  std::string end_point = segment_id->first_node("Segment_end_point")->value();
	  std::vector<double> start_vec, end_vec;
	  double conversion_result;
	  std::istringstream is_start(start_point), is_end(end_point);
	  while(is_start >> conversion_result)
		  start_vec.push_back(conversion_result);
	  while(is_end>>conversion_result)
		  end_vec.push_back(conversion_result);
	  for (unsigned int coord = 0; coord < dim; coord++)
	  {
		  segment_start_point[coord] = start_vec[coord];
		  segment_end_point[coord] = end_vec[coord];
	  }
  }


//******************* mechanical member and functions *********************
  template<int dim>
  void HeatEquation<dim>::setup_mech_system()
  {
	  dof_handler_disp.distribute_dofs (fe_collection_disp);
//	  pcout<<"setup_mech_system()---00"<< std::endl;
	  locally_owned_dofs_disp = dof_handler_disp.locally_owned_dofs ();
//	  pcout<<"setup_mech_system()---000"<< std::endl;
	  DoFTools::extract_locally_relevant_dofs (dof_handler_disp,
													 locally_relevant_dofs_disp);
//pcout<<"setup_mech_system()---0"<< std::endl;
//	  typename Triangulation<dim>::active_cell_iterator
//							cell = triangulation.begin_active(),
//							endc = triangulation.end();
//	  for (; cell != endc; ++cell)
//	  {
//		  if (cell->is_locally_owned())
//		  {
//			  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//			  {
//				  if (cell->face(f)->at_boundary())
//				  {
//					  const Point<dim> face_center = cell->face(f)->center();
//					  if (fabs(face_center[dim - 1] + EquationData::g_base_height) < 1e-6)
//					  {	// bottom surface, Dirchlet boundary
////						  pcout<<"get boudary id: "<<cell->face(f)->boundary_id ()<<endl;
//						  cell->face(f)->set_boundary_id (2);
//					  }
////					  else
////					  {
////						  cell->face(f)->set_all_boundary_ids(0);
////					  }
//				  }
//			  }
//		  }
//	  }
	  mesh_changed_flg_for_mech = true;

//	  pcout<<"setup_mech_system()---1"<< std::endl;
	  /* setup hanging nodes and Dirichlet constraints */
	  {
		  constraints_disp.clear ();
		  constraints_disp.reinit(locally_relevant_dofs_disp);
		  DoFTools::make_hanging_node_constraints (dof_handler_disp,
	                                             constraints_disp);
		  constraints_disp.close();
		  compute_dirichlet_constraints();
	  }

  	  DynamicSparsityPattern sparsity_pattern_disp(locally_relevant_dofs_disp);
  	  DoFTools::make_sparsity_pattern(dof_handler_disp,
  			  	  	  	  	  	  	    sparsity_pattern_disp,
										constraints_dirichlet_and_hanging_nodes,
  	                                    /*keep_constrained_dofs = */ false);
	  SparsityTools::distribute_sparsity_pattern (sparsity_pattern_disp,
			  	  	  	  	  	  	  	  	  	  	dof_handler_disp.n_locally_owned_dofs_per_processor(),
	                                                mpi_communicator,
	                                                locally_relevant_dofs_disp);

	  newton_matrix_disp.reinit (locally_owned_dofs_disp,
	                          locally_owned_dofs_disp,
	                          sparsity_pattern_disp,
	                          mpi_communicator);
	  newton_rhs_disp.reinit(locally_owned_dofs_disp, mpi_communicator);

  	  if (timestep_number == 0) // || current_refinement_cycle!=0)
  	  {
  		  solution_disp.reinit (dof_handler_disp.n_dofs());
  	  }
  	  incremental_displacement.reinit(dof_handler_disp.n_dofs());

  	  fraction_of_plastic_q_points_per_cell.reinit (triangulation.n_active_cells());
  }

  template <int dim>
   void HeatEquation<dim>::compute_dirichlet_constraints()
   {
	  constraints_dirichlet_and_hanging_nodes.clear();
	  //added -05/28 -start
	  constraints_dirichlet_and_hanging_nodes.reinit(locally_relevant_dofs_disp);
	  // added -05/28 -end
	  constraints_dirichlet_and_hanging_nodes.merge(constraints_disp);
	  std::vector<bool> component_mask(dim);
	  component_mask[0] = true;
	  component_mask[1] = true;
	  component_mask[2] = true;
	  VectorTools::interpolate_boundary_values (dof_handler_disp,
	                                                  2,	// bottom surface of substrate
													  ZeroFunction<dim>(dim),
	                                                  constraints_dirichlet_and_hanging_nodes,
	                                                  component_mask);
	  constraints_dirichlet_and_hanging_nodes.close();
   }

  template <int dim>
   void HeatEquation<dim>::setup_quadrature_point_history ()
   {
//     unsigned int our_cells = 0;
//     for (typename Triangulation<dim>::active_cell_iterator
//          cell = triangulation.begin_active();
//          cell != triangulation.end(); ++cell)
//       if (cell->is_locally_owned())
//         ++our_cells;
//cout<<"our_cells = "<<our_cells <<", in CPU-"<<this_mpi_process<<endl;
//cout<<"locally owned cells = "<<triangulation.n_locally_owned_active_cells() <<", in CPU-"<<this_mpi_process<<endl;
     triangulation.clear_user_data();

     {
       std::vector<PointHistory<dim> > tmp;
       tmp.swap (quadrature_point_history);
     }
     quadrature_point_history.resize (triangulation.n_locally_owned_active_cells() * //our_cells *
    		 	 	 	 	 	 	 quadrature_collection_disp[0].size());

     unsigned int history_index = 0;
     for (typename Triangulation<dim>::active_cell_iterator
          cell = triangulation.begin_active();
          cell != triangulation.end(); ++cell)
       if (cell->is_locally_owned())
         {
           cell->set_user_pointer (&quadrature_point_history[history_index]);
           history_index += quadrature_collection_disp[0].size();
         }

     Assert (history_index == quadrature_point_history.size(),
             ExcInternalError());
   }

  template <int dim>
  void HeatEquation<dim>::update_quadrature_point_history ()
  {
//    hp::FEValues<dim> hp_fe_values (fe_collection_disp, quadrature_collection_disp,
//                             update_values | update_gradients | update_quadrature_points);
//	  hp::FEValues<dim> hp_fe_temp_values (fe_collection, quadrature_collection,
//			  	  	  	  	  update_values   | update_gradients |
//							  update_quadrature_points | update_JxW_values);
//
//    std::vector<std::vector<Tensor<1,dim> > > displacement_increment_grads (quadrature_collection_disp[0].size(),
//   		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 std::vector<Tensor<1,dim> >(dim));
//    std::vector<SymmetricTensor<2, dim> > incremental_strain_tensor(quadrature_collection_disp[0].size());
//
//    std::vector<double> cell_deltatemp_values(quadrature_collection[0].size());
//    std::vector<double> old_cell_deltatemp_values(quadrature_collection[0].size());

    typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  	  	  	  	  cell = dof_handler_disp.begin_active(),
									  endc = dof_handler_disp.end();
//    typename hp::DoFHandler<dim>::active_cell_iterator cell_heat = dof_handler.begin_active();

    unsigned int cnt_cells (0);
    const FEValuesExtractors::Vector displacement(0);
    for (; cell != endc; ++cell, /*++cell_heat,*/ ++cnt_cells)
    {
    	if (cell->is_locally_owned())
    	{
    	unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
    	if (dofs_per_cell != 0)
    	{
    		PointHistory<dim> * local_quadrature_points_history
					 = reinterpret_cast<PointHistory<dim> *> (cell->user_pointer());
    		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),
					  ExcInternalError());
    		Assert (local_quadrature_points_history < &quadrature_point_history.back(),
					  ExcInternalError());

    		for (unsigned int q_point = 0; q_point < quadrature_collection_disp[0].size(); ++q_point)
    		{
    			local_quadrature_points_history[q_point].pre_stress = local_quadrature_points_history[q_point].old_stress;
    			local_quadrature_points_history[q_point].pre_plastic_strain = local_quadrature_points_history[q_point].old_plastic_strain;
    			local_quadrature_points_history[q_point].pre_effective_plastic_strain = local_quadrature_points_history[q_point].old_effective_plastic_strain;
//    			local_quadrature_points_history[q_point].point = fe_values.get_quadrature_points ()[q_point];
    		}
    	}
    	}
    }
  }

  template <int dim>
  SymmetricTensor<2,dim>
  HeatEquation<dim>::get_thermal_strain (const  double/*Vector<double>*/ &deltatemp, const double &thermoexpan)         //, const double &Tref, const double &alpha_value)
  {
//	  Assert (grad.size() == dim, ExcInternalError());

	  SymmetricTensor<2,dim> strain;
	  for (unsigned int i=0; i<dim; ++i)
		  strain[i][i] = thermoexpan*(deltatemp - Temp_ref);

	  for (unsigned int i=0; i<dim; ++i)
		  for (unsigned int j=i+1; j<dim; ++j)
			  strain[i][j] = 0;

	  return strain;
  }

  template<int dim>
  void HeatEquation<dim>::output_mech_results() //const
  {
	  TimerOutput::Scope t(computing_timer, "Graphical output-Mech");

	  DataOut<dim,hp::DoFHandler<dim> > data_out;
	  data_out.attach_dof_handler (dof_handler_disp);

	    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
	    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
	    data_out.add_data_vector(solution_disp,
	                             std::vector<std::string> (dim, "displacement"),
	                             DataOut<dim, hp::DoFHandler<dim>>::type_dof_data, data_component_interpretation);

	  std::vector<std::string> solution_names;
	  switch (dim)
	  {
	  	  case 1:
	  		  solution_names.push_back ("displacement");
	  		  break;
	  	  case 2:
	  		  solution_names.push_back ("x_displacement");
	  		  solution_names.push_back ("y_displacement");
	  		  break;
	      case 3:
	    	  solution_names.push_back ("x_displacement");
	    	  solution_names.push_back ("y_displacement");
	    	  solution_names.push_back ("z_displacement");
	    	  break;
	      default:
	    	  Assert (false, ExcNotImplemented());
	  }
	  data_out.add_data_vector (FE_Type, "FE_Type");
	  data_out.add_data_vector (cell_material, "cell_material");
	  data_out.add_data_vector (solution_disp, solution_names);
//	  data_out.build_patches ();
//	  data_out.write_vtk (output);

	  Vector<double> norm_of_stress (triangulation.n_active_cells());
	  {
		  typename Triangulation<dim>::active_cell_iterator
		  cell = triangulation.begin_active(),
	      endc = triangulation.end();
		  for (; cell!=endc; ++cell)
			  if (cell->is_locally_owned())
			  {
				  SymmetricTensor<2,dim> accumulated_stress;
				  for (unsigned int q = 0; q < quadrature_collection_disp[0].size(); ++q)
					  accumulated_stress += reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].old_stress;

				  norm_of_stress(cell->active_cell_index()) = (accumulated_stress / quadrature_collection_disp[0].size()).norm();
//				  cout<<"norm stress("<<cell->active_cell_index()<<"): "<<norm_of_stress(cell->active_cell_index())<<std::endl;
			  }
			  else
				  norm_of_stress(cell->active_cell_index()) = -1e+20;
	  }
	  data_out.add_data_vector (norm_of_stress, "norm_of_stress");
	  data_out.add_data_vector (fraction_of_plastic_q_points_per_cell, "fraction_of_plastic_q_points");

	  Vector<double> norm_of_strain (triangulation.n_active_cells());
	  {
		  typename Triangulation<dim>::active_cell_iterator
		  cell = triangulation.begin_active(),
	      endc = triangulation.end();
		  for (; cell!=endc; ++cell)
			  if (cell->is_locally_owned())
			  {
				  SymmetricTensor<2,dim> accumulated_strain;
				  for (unsigned int q = 0; q < quadrature_collection_disp[0].size(); ++q)
					  accumulated_strain += reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].old_strain;

				  norm_of_strain(cell->active_cell_index()) = (accumulated_strain / quadrature_collection_disp[0].size()).norm();
//				  cout<<"norm stress("<<cell->active_cell_index()<<"): "<<norm_of_stress(cell->active_cell_index())<<std::endl;
			  }
			  else
				  norm_of_strain(cell->active_cell_index()) = -1e+20;
	  }
	  data_out.add_data_vector (norm_of_strain, "norm_of_strain");

	  data_out.build_patches ();

	  std::string filename = "Mechanical-solution-"
	    						+ Utilities::int_to_string(timestep_number, 3) +
								"." +
								Utilities::int_to_string(dof_handler_disp.get_triangulation().locally_owned_subdomain(), 4);
	  filename = output_mech_dir + filename;
	  std::ofstream output((filename + ".vtu").c_str());
	  data_out.write_vtu(output);
	  if (this_mpi_process == 0)
	  {
		  std::vector<std::string> filenames;
		  for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
			  filenames.push_back ("Mechanical-solution-" +
													  Utilities::int_to_string (timestep_number, 3) +
													  "." +
													  Utilities::int_to_string (i, 4) +
													  ".vtu");
		  std::ofstream master_output ((output_mech_dir + "Mechanical-solution-" +
													  Utilities::int_to_string (timestep_number, 3) +
													  ".pvtu").c_str());
		  data_out.write_pvtu_record (master_output, filenames);  // defaulty, write_pvtu_record() function will search the filenames in the current folder.
	  }


	  SymmetricTensor<2, dim> stress_at_qpoint,
	  	  	  	  	  	  	  	  	  	  	  	  strain_at_qpoint;	// plastic strain

	  FE_DGQ<dim> history_fe (1);
	  DoFHandler<dim> history_dof_handler (triangulation);
	  history_dof_handler.distribute_dofs (history_fe);
	  std::vector< std::vector< Vector<double> > >
	  	  	  	  	  	  history_stress_field (dim, std::vector< Vector<double> >(dim)),
						  local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
						  local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim)),
	  	  	  	  	  	  history_strain_field (dim, std::vector< Vector<double> >(dim)),
						  local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
						  local_history_strain_fe_values (dim, std::vector< Vector<double> >(dim));
	  for (unsigned int i=0; i<dim; i++)
		  for (unsigned int j=0; j<dim; j++)
		  {
			  history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
			  local_history_stress_values_at_qpoints[i][j].reinit(quadrature_collection_disp[0].size());	// for dim = 3, quadrature_formula.size = 8
			  local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);

			  history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
			  local_history_strain_values_at_qpoints[i][j].reinit(quadrature_collection_disp[0].size());
			  local_history_strain_fe_values[i][j].reinit(history_fe.dofs_per_cell);
		  }

	  Vector<double>  VM_stress_field (history_dof_handler.n_dofs()),
			  	  	  	  	  	  local_VM_stress_values_at_qpoints (quadrature_collection_disp[0].size()),
								  local_VM_stress_fe_values (history_fe.dofs_per_cell),

								  VM_strain_field (history_dof_handler.n_dofs()),
								  effective_plastic_strain_field (history_dof_handler.n_dofs()),
								  local_VM_strain_values_at_qpoints (quadrature_collection_disp[0].size()),// plastic strain (tensor)
								  local_effective_plastic_strain_values_at_qpoints (quadrature_collection_disp[0].size()), // effective plastic strain (scalar)
								  local_VM_strain_fe_values (history_fe.dofs_per_cell),
								  local_effective_plastic_strain_fe_values(history_fe.dofs_per_cell);

	  FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell, quadrature_collection_disp[0].size());
	  FETools::compute_projection_from_quadrature_points_matrix (history_fe,
			  	  	  	  	  	  	  	  	  	  	  	  	  quadrature_collection_disp[0], quadrature_collection_disp[0],
															  qpoint_to_dof_matrix);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  	  	  	  	  	  cell = dof_handler_disp.begin_active(),
										  endc = dof_handler_disp.end();
	  typename DoFHandler<dim>::active_cell_iterator
										  dg_cell = history_dof_handler.begin_active();

	  const FEValuesExtractors::Vector displacement(0);

	  for (; cell!=endc; ++cell, ++dg_cell)
		  if (cell->is_locally_owned())
		  {
			  PointHistory<dim> *local_quadrature_points_history
			  	  	  	  	  	  	  	  	  	  = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
			  Assert (local_quadrature_points_history >=
					  	  	  	  &quadrature_point_history.front(),
								  ExcInternalError());
			  Assert (local_quadrature_points_history <
					  	  	  	  &quadrature_point_history.back(),
								  ExcInternalError());

			  for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
			  {
				  stress_at_qpoint = local_quadrature_points_history[q].old_stress;
				  strain_at_qpoint = local_quadrature_points_history[q].old_plastic_strain;
//				  SymmetricTensor<4, dim> stress_strain_tensor;
//				  constitutive_law.get_stress_strain_tensor(local_quadrature_points_history[q].old_strain,
//				                                                          stress_strain_tensor);
//				  strain_at_qpoint = local_quadrature_points_history[q].old_strain
//				  						  	  	  - inverse_elastic_stress_strain_tensor*
//				  								  stress_strain_tensor*local_quadrature_points_history[q].old_strain; // plastic strain = total strain - inverse(elastic stress strain tensor)*stress strain tensor*total stress strain

				  for (unsigned int i=0; i<dim; i++)
					  for (unsigned int j=i; j<dim; j++)
					  {
						  local_history_stress_values_at_qpoints[i][j](q) = stress_at_qpoint[i][j];
						  local_history_strain_values_at_qpoints[i][j](q) = strain_at_qpoint[i][j];
					  }

				  local_VM_stress_values_at_qpoints(q) = Evaluation::get_von_Mises_stress(stress_at_qpoint);
				  local_VM_strain_values_at_qpoints(q) = Evaluation::get_von_Mises_stress(strain_at_qpoint);
				  local_effective_plastic_strain_values_at_qpoints(q) = local_quadrature_points_history[q].old_effective_plastic_strain;
			  }
			  for (unsigned int i=0; i<dim; i++)
				  for (unsigned int j=i; j<dim; j++)
				  {
					  qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
							  	  	  	  	  	  	  	  	  	  	  local_history_stress_values_at_qpoints[i][j]);
					  dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
							  	  	  	  	  	  	  	  	  	  	  history_stress_field[i][j]);
					  qpoint_to_dof_matrix.vmult (local_history_strain_fe_values[i][j],
							  	  	  	  	  	  	  	  	  	  	  local_history_strain_values_at_qpoints[i][j]);
					  dg_cell->set_dof_values (local_history_strain_fe_values[i][j],
							  	  	  	  	  	  	  	  	  	  	  history_strain_field[i][j]);
				  }

			  qpoint_to_dof_matrix.vmult (local_VM_stress_fe_values,
					  	  	  	  	  	  	  	  	  	  	  local_VM_stress_values_at_qpoints);
			  dg_cell->set_dof_values (local_VM_stress_fe_values,
					  	  	  	  	  	  	  	  	  	  VM_stress_field);
			  qpoint_to_dof_matrix.vmult (local_VM_strain_fe_values,
					  	  	  	  	  	  	  	  	  	  	  local_VM_strain_values_at_qpoints);
			  dg_cell->set_dof_values (local_VM_strain_fe_values,
					  	  	  	  	  	  	  	  	  	  VM_strain_field);
			  qpoint_to_dof_matrix.vmult (local_effective_plastic_strain_fe_values,
					  	  	  	  	  	  	  	  	  	  	  local_effective_plastic_strain_values_at_qpoints);
			  dg_cell->set_dof_values (local_effective_plastic_strain_fe_values,
					  	  	  	  	  	  	  	  	  	  effective_plastic_strain_field);
		  }



//	  FE_Q<dim>          fe_1 (1);
//	  DoFHandler<dim>    dof_handler_1 (triangulation);
//	  dof_handler_1.distribute_dofs (fe_1);
//
//	  AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
//			  	  	  	  	  ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));
//
//	  std::vector< std::vector< Vector<double> > > history_stress_on_vertices (dim, std::vector< Vector<double> >(dim)),
//			  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	history_strain_on_vertices (dim, std::vector< Vector<double> >(dim));
//	  for (unsigned int i=0; i<dim; i++)
//		  for (unsigned int j=0; j<dim; j++)
//		  {
//			  history_stress_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
//			  history_strain_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
//		  }
//
//	  Vector<double>  VM_stress_on_vertices (dof_handler_1.n_dofs()),
//			  	  	  	  	  	  VM_strain_on_vertices (dof_handler_1.n_dofs()),
//								  counter_on_vertices (dof_handler_1.n_dofs());
//	  VM_stress_on_vertices = 0;
//	  VM_strain_on_vertices = 0;
//	  counter_on_vertices = 0;
//
//	  cell = dof_handler_disp.begin_active();
//	  dg_cell = history_dof_handler.begin_active();
//	  typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
//	  for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
//		  if (cell->is_locally_owned())
//		  {
//			  dg_cell->get_dof_values (VM_stress_field, local_VM_stress_fe_values);
//			  dg_cell->get_dof_values (VM_strain_field, local_VM_strain_fe_values);
//			  for (unsigned int i=0; i<dim; i++)
//				  for (unsigned int j=0; j<dim; j++)
//				  {
//					  dg_cell->get_dof_values (history_stress_field[i][j],
//							  	  	  	  	  local_history_stress_fe_values[i][j]);
//					  dg_cell->get_dof_values (history_strain_field[i][j],
//							  	  	  	  	  local_history_strain_fe_values[i][j]);
//				  }
//
//			  for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//			  {
//				  types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);
//				  counter_on_vertices (dof_1_vertex) += 1;
//				  VM_stress_on_vertices (dof_1_vertex) += local_VM_stress_fe_values (v);
//				  VM_strain_on_vertices (dof_1_vertex) += local_VM_strain_fe_values (v);
//				  for (unsigned int i=0; i<dim; i++)
//					  for (unsigned int j=0; j<dim; j++)
//					  {
//						  history_stress_on_vertices[i][j](dof_1_vertex) +=
//								  	  	  local_history_stress_fe_values[i][j](v);
//						  history_strain_on_vertices[i][j](dof_1_vertex) +=
//								  	  	  local_history_strain_fe_values[i][j](v);
//					  }
//			  }
//		  }
//
//	  for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
//	  {
//		  VM_stress_on_vertices(id) /= counter_on_vertices(id);
//		  VM_strain_on_vertices(id) /= counter_on_vertices(id);
//		  for (unsigned int i=0; i<dim; i++)
//			  for (unsigned int j=0; j<dim; j++)
//			  {
//				  history_stress_on_vertices[i][j](id) /= counter_on_vertices(id);
//				  history_strain_on_vertices[i][j](id) /= counter_on_vertices(id);
//			  }
//	  }

//	  if (show_stresses)
	  {
//		  std::string filename_2 = "Stress-solution-"
//		    						+ Utilities::int_to_string(timestep_number, 3) +
//	                ".vtk";
//		  std::ofstream output_2 (filename_2.c_str());

//		  {
			  DataOut<dim>  data_out_2;
			  data_out_2.attach_dof_handler (history_dof_handler);

			  data_out_2.add_data_vector (history_stress_field[0][0], "stress_xx");
			  data_out_2.add_data_vector (history_stress_field[1][1], "stress_yy");
			  data_out_2.add_data_vector (history_stress_field[0][1], "stress_xy");
			  data_out_2.add_data_vector (VM_stress_field, "Von_Mises_stress");
			  if (dim == 3)
			  {
				  data_out_2.add_data_vector (history_stress_field[0][2], "stress_xz");
				  data_out_2.add_data_vector (history_stress_field[1][2], "stress_yz");
				  data_out_2.add_data_vector (history_stress_field[2][2], "stress_zz");
			  }

			  data_out_2.add_data_vector (history_strain_field[0][0], "plastic_strain_xx");
			  data_out_2.add_data_vector (history_strain_field[1][1], "plastic_strain_yy");
			  data_out_2.add_data_vector (history_strain_field[0][1], "plastic_strain_xy");
			  data_out_2.add_data_vector (VM_strain_field, "Von_Mises_strain_plastic");
			  data_out_2.add_data_vector (effective_plastic_strain_field, "Effective_plastic_strain");
			  if (dim == 3)
			  {
				  data_out_2.add_data_vector (history_strain_field[0][2], "plastic_strain_xz");
				  data_out_2.add_data_vector (history_strain_field[1][2], "plastic_strain_yz");
				  data_out_2.add_data_vector (history_strain_field[2][2], "plastic_strain_zz");
			  }
//		  }

////		  {
//			  data_out_2.attach_dof_handler (dof_handler_1);
//
//			  data_out_2.add_data_vector (history_stress_on_vertices[0][0], "stress_xx_averaged");
//			  data_out_2.add_data_vector (history_stress_on_vertices[1][1], "stress_yy_averaged");
//			  data_out_2.add_data_vector (history_stress_on_vertices[0][1], "stress_xy_averaged");
//			  data_out_2.add_data_vector (VM_stress_on_vertices, "Von_Mises_stress_averaged");
//			  if (dim == 3)
//			  {
//				  data_out_2.add_data_vector (history_stress_on_vertices[0][2], "stress_xz_averaged");
//				  data_out_2.add_data_vector (history_stress_on_vertices[1][2], "stress_yz_averaged");
//				  data_out_2.add_data_vector (history_stress_on_vertices[2][2], "stress_zz_averaged");
//			  }
//
////		  }

			  data_out_2.add_data_vector (FE_Type, "FE_Type");
			  data_out_2.add_data_vector (cell_material, "cell_material");

			  data_out_2.build_patches ();
//			  data_out_2.write_vtk(output_2);
			  std::string filename_2 = "Stress-solution-"
			    						+ Utilities::int_to_string(timestep_number, 3) +
										"."+
										 Utilities::int_to_string
										 (dof_handler_disp.get_triangulation().locally_owned_subdomain(), 4);
			  filename_2 = output_mech_dir + filename_2;
			  std::ofstream output_2((filename_2 + ".vtu").c_str());
			  data_out_2.write_vtu(output_2);
			  if (this_mpi_process == 0)
			  {
				  std::vector<std::string> filenames;
				  for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
					  filenames.push_back ("Stress-solution-" +
															  Utilities::int_to_string (timestep_number, 3) +
															  "." +
															  Utilities::int_to_string (i, 4) +
															  ".vtu");
				  std::ofstream master_output ((output_mech_dir + "Stress-solution-" +
															  Utilities::int_to_string (timestep_number, 3) +
															  ".pvtu").c_str());
				  data_out_2.write_pvtu_record (master_output, filenames);  // defaulty, write_pvtu_record() function will search the filenames in the current folder.
			  }
	  }




	  // table results
	  const Point<dim> point_A(0, 0, 0.05e-3); //(-0.00937143, 0, 0); //(-0.0114286, 0, 0);  //(-0.0121143, 0, 0);
	  Vector<double>   disp_A(dim);		//(dim);
	  Vector<double>   copy_solution(solution_disp);

	  Evaluation::PointValuesEvaluation<dim> point_values_evaluation(point_A);
	  point_values_evaluation.compute (dof_handler_disp, copy_solution, disp_A);

	  {
		  if (this_mpi_process == 0)
		  {
//				  table_results.set_auto_fill_mode(true);
			  table_results_4.add_value("time", time);
			  table_results_4.set_precision("time", 7);
		  }
		  table_results_4.add_value("u_Ax", disp_A(0));
		  table_results_4.set_precision("u_Ax", 11);
		  table_results_4.add_value("u_Ay", disp_A(1));
		  table_results_4.set_precision("u_Ay", 11);
		  table_results_4.add_value("u_Az", disp_A(2));
		  table_results_4.set_precision("u_Az", 11);

		  std::string filename_2 = "Results_disp_A_" + Utilities::int_to_string(dof_handler_disp.get_triangulation().locally_owned_subdomain(), 4);
		  filename_2 = output_mech_dir + filename_2;
		  std::ofstream output_txt((filename_2 + ".txt").c_str());
		  table_results_4.write_text(output_txt);
	  }


	  // table_5 compute vm stress of selected point
	  {
//		  const unsigned int dofs_per_vertex = dof_handler_disp.get_fe_collection()[0].dofs_per_vertex;
		  double point_value_VM_stress = 1e20;
//		  double point_values = 1e20;

		  typename hp::DoFHandler<dim>::active_cell_iterator
	  			cell = dof_handler_disp.begin_active(),
	  			endc = dof_handler_disp.end();
		  bool evaluation_point_found = false;
		  for (; (cell!=endc) && !evaluation_point_found; ++cell)
		  {
			  const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
			  if (dofs_per_cell != 0)
			  {
				  if (cell->is_locally_owned() && !evaluation_point_found)
					  for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex)
					  {
						  if (cell->vertex(vertex).distance (point_A)
	  								<
									cell->diameter() * 1e-1)
						  {
							  PointHistory<dim> *local_quadrature_points_history
	  			                  = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
							  Assert (local_quadrature_points_history >=
	  			                        &quadrature_point_history.front(),
	  			                        ExcInternalError());
							  Assert (local_quadrature_points_history <
	  			                        &quadrature_point_history.back(),
	  			                        ExcInternalError());

							  // Then loop over the quadrature points of this cell:
							  double min_distance = 1e10;
							  unsigned int min_q_index = 0;
							  for (unsigned int q=0; q<quadrature_collection_disp[0].size(); ++q)
							  {
								  double tmp_distance = cell->vertex(vertex).distance (local_quadrature_points_history[q].point);
								  if (tmp_distance < min_distance)
								  {
									  min_distance = tmp_distance;
									  min_q_index = q;
								  }
							  }

							  point_value_VM_stress = Evaluation::get_von_Mises_stress(local_quadrature_points_history[min_q_index].pre_stress);
//							  point_values = VM_stress;

							  evaluation_point_found = true;
							  break;
						  }
					  }
			  }
		  }

		  if (this_mpi_process == 0)
		  {
//				  table_results.set_auto_fill_mode(true);
			  table_results_5.add_value("time", time);
			  table_results_5.set_precision("time", 7);
		  }
		  table_results_5.add_value("VM_point_A", point_value_VM_stress);
//		  table_results_5.set_precision("u_Ax", 11);

		  std::string filename_2 = "Results_VM_A_" + Utilities::int_to_string(dof_handler_disp.get_triangulation().locally_owned_subdomain(), 4);
		  filename_2 = output_mech_dir + filename_2;
		  std::ofstream output_txt((filename_2 + ".txt").c_str());
		  table_results_5.write_text(output_txt);
	  } // table 5 end

  }


  template <int dim>
  void HeatEquation<dim>::move_mesh (bool distort_flag)
  {
//	  (distort_flag == true) ? std::cout << "    Moving mesh..." << std::endl : std::cout << "    Moving back mesh..." << std::endl;

	  std::vector<bool> vertex_touched (triangulation.n_vertices(), false);
	  for (typename hp::DoFHandler<dim>::active_cell_iterator
			  	  	  cell = dof_handler_disp.begin_active ();
			  	  	  cell != dof_handler_disp.end(); ++cell)
	  {
		  if (cell->is_locally_owned())
		  {
		  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
		  if (dofs_per_cell != 0)
		  {
			  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
				  if (vertex_touched[cell->vertex_index(v)] == false)
				  {
					  vertex_touched[cell->vertex_index(v)] = true;
					  Point<dim> vertex_displacement;
					  for (unsigned int d=0; d<dim; ++d)
					  {
//						  std::cout << "    vertex_dof_index(v,d) = "<<cell->vertex_dof_index(v,d) << std::endl;
						  vertex_displacement[d]
											  = solution_disp(cell->vertex_dof_index(v,d,0));
//						  std::cout << "    vertex_displacement = "<<vertex_displacement[d] << std::endl;
					  }
					  // scale factor
					  vertex_displacement *= 1;
					  (distort_flag == true) ?cell->vertex(v) += vertex_displacement : cell->vertex(v) -= vertex_displacement;
				  }
		  }
		  }
	  }
  }

  template <int dim>
  void
  HeatEquation<dim>:: calculate_stress_strain_for_integration_points(unsigned int i, const Vector<double> &old_solution_disp)
  {
	  TimerOutput::Scope t(computing_timer, "Calculate stress and strain");

	  hp::FEValues<dim> hp_fe_values(fe_collection_disp, quadrature_collection_disp,
			  	  	  	  	  update_values | update_gradients |
							  update_quadrature_points | update_JxW_values);

	  hp::FEValues<dim> hp_fe_temp_values (fe_collection, quadrature_collection,
			  	  	  	  	  update_values   | update_gradients |
							  update_quadrature_points | update_JxW_values);

	  std::vector<double> cell_temp_values(quadrature_collection[0].size()), old_cell_temp_values(quadrature_collection[0].size());

//	  const unsigned int dofs_per_active_cell   = fe_collection_disp[0].dofs_per_cell;
	  const unsigned int n_q_points      = quadrature_collection_disp[0].size();

//	  std::vector<types::global_dof_index>   local_dof_indices(dofs_per_active_cell);
	  std::vector<SymmetricTensor<2, dim> > strain_tensors_i_t_delta_t(n_q_points), old_strain_tensors_t(n_q_points);

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  	  	  	  	  cell = dof_handler_disp.begin_active(),
									  endc = dof_handler_disp.end();
	  typename hp::DoFHandler<dim>::active_cell_iterator cell_heat = dof_handler.begin_active();

	  double melt_point = parameters.melt_point;
	  unsigned int cnt_cells (0);
	  const FEValuesExtractors::Vector displacement(0);
	  for (; cell != endc; ++cell, ++cell_heat, ++cnt_cells)
	  {
		  if (cell->is_locally_owned())
		  {
		  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
		  if (dofs_per_cell != 0)
		  {
			  hp_fe_values.reinit(cell);
			  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

			  hp_fe_temp_values.reinit(cell_heat);
			  const FEValues<dim> &fe_temp_values = hp_fe_temp_values.get_present_fe_values();
			  fe_temp_values.get_function_values(solution, cell_temp_values);
			  fe_temp_values.get_function_values(old_solution, old_cell_temp_values);

			  fe_values[displacement].get_function_symmetric_gradients(solution_disp, strain_tensors_i_t_delta_t);
			  fe_values[displacement].get_function_symmetric_gradients(old_solution_disp, old_strain_tensors_t);


			  PointHistory<dim> *local_quadrature_points_history = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
			  Assert (local_quadrature_points_history >= &quadrature_point_history.front(),
					  ExcInternalError());
			  Assert (local_quadrature_points_history < &quadrature_point_history.back(),
					  ExcInternalError());
			  // temperature dependent material
			  std::vector<double> material_temperature_list(16);
			  material_temperature_list[0] = 293; material_temperature_list[1] = 373;
			  for (unsigned int iter = 2; iter < 15; iter++)
				  material_temperature_list[iter] = material_temperature_list[iter - 1] + 100;
			  material_temperature_list[15] = 1723;
			  // consistent cell material type
			  double consistent_cell_material_type = 0;
			  if (fabs(old_cell_material[cnt_cells] - cell_material[cnt_cells]) < 1e-3)
				  consistent_cell_material_type = cell_material[cnt_cells];
			  else
			  {
				  if(old_cell_material[cnt_cells] == 2 && cell_material[cnt_cells] != 2)// 2 (powder) -> 1 (liquid): 2   2 (powder) -> 0 (solid): 2
					  consistent_cell_material_type = old_cell_material[cnt_cells];
				  else if(old_cell_material[cnt_cells] == 0 && cell_material[cnt_cells] != 0) //0 (solid) -> 1 (liquid): 1      0 (solid) -> 2 (powder): 0
					  consistent_cell_material_type = old_cell_material[cnt_cells];
				  else if(old_cell_material[cnt_cells] == 1 && cell_material[cnt_cells] == 0) //1 (liquid) -> 0 (solid): 0
					  consistent_cell_material_type = cell_material[cnt_cells];
				  else if(old_cell_material[cnt_cells] == 1 && cell_material[cnt_cells] == 2) // 1 (liquid) -> 2 (powder): 1   --- not possible to happen
					  consistent_cell_material_type = old_cell_material[cnt_cells];
			  }

			  double substep_size;	// delta_tao
			  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			  {	// loop for each integration point
//				  if (old_cell_temp_values[q_point] == cell_temp_values[q_point]) // even though old temperature == current temperature, the old strain may not be equal to current strain.
//					  continue;

				  std::vector<double> input_temperature_list;
				  input_temperature_list.push_back(old_cell_temp_values[q_point]);
				  if (i > 0)
				  {// when i = 0, only loop for one segment
					  if (old_cell_temp_values[q_point] < cell_temp_values[q_point]) // old temperature < current temperature
						  for (typename std::vector<double>::iterator iter_temp = material_temperature_list.begin();
														  iter_temp != material_temperature_list.end(); iter_temp++)
						  {// store the temperature between old_cell_temp and current_cell_temp
							  if (old_cell_temp_values[q_point] < *iter_temp && *iter_temp < cell_temp_values[q_point])
								  input_temperature_list.push_back(*iter_temp);
						  }
					  else // old temperature >= current temperature
						  for (typename std::vector<double>::iterator iter_temp = material_temperature_list.end();
														  iter_temp != material_temperature_list.begin(); iter_temp--)
						  {
							  if (cell_temp_values[q_point] < *iter_temp && *iter_temp < old_cell_temp_values[q_point])
								  input_temperature_list.push_back(*iter_temp);
						  }
				  }
				  input_temperature_list.push_back(cell_temp_values[q_point]);

				  unsigned int j_iter = 1;  // total number of substeps
				  double start_temp = input_temperature_list[0], end_temp;
				  unsigned int iter = 1;
				  while (start_temp != input_temperature_list[input_temperature_list.size() - 1])
				  {// iterate over each segment.  start_temp --> end_temp
					  end_temp = input_temperature_list[iter]; // update: end_temp

					  // set parameters at segment start time
					  constitutive_law.set_current_temperature(start_temp, consistent_cell_material_type);  //(old_cell_temp_values[q_point], old_cell_material[cnt_cells]);
					  double old_thermoexpan = constitutive_law.get_thermal_expansion(), old_miu = constitutive_law.get_mu(), old_K = constitutive_law.get_K(),
								  old_H = constitutive_law.get_hardening_parameter(), old_sigma_y0 = constitutive_law.get_sigma_y0();
//								  old_derivative_sigma_y0 = constitutive_law.get_derivative_sigma_y0(),
//								  old_derivative_H = constitutive_law.get_derivative_hardening_parameter();
					  SymmetricTensor<2,dim> old_thermal_strain_qpoint = get_thermal_strain(start_temp, old_thermoexpan);
					  SymmetricTensor<4, dim> old_elastic_stress_strain_tensor;
					  constitutive_law.get_elastic_stress_strain_tensor(old_elastic_stress_strain_tensor);
					  // set parameters at segment end time
					  constitutive_law.set_current_temperature(end_temp, consistent_cell_material_type);   //(cell_temp_values[q_point], cell_material[cnt_cells]);
					  double thermoexpan = constitutive_law.get_thermal_expansion(), miu = constitutive_law.get_mu(), K = constitutive_law.get_K(),
								  H = constitutive_law.get_hardening_parameter(), sigma_y0 = constitutive_law.get_sigma_y0();
//								  derivative_sigma_y0 = constitutive_law.get_derivative_sigma_y0(),
//								  derivative_H = constitutive_law.get_derivative_hardening_parameter();//,
//								  dot_miu = (miu - old_miu)/time_step;   // it seems to be wrong, the time_step should be divided by q. ---03/22/19
					  SymmetricTensor<2,dim> thermal_strain_qpoint = get_thermal_strain(end_temp, thermoexpan);
					  SymmetricTensor<2,dim> delta_thermal_strain_qpoint = thermal_strain_qpoint - old_thermal_strain_qpoint;
					  SymmetricTensor<4, dim> elastic_stress_strain_tensor;
					  constitutive_law.get_elastic_stress_strain_tensor(elastic_stress_strain_tensor);
					  SymmetricTensor<4, dim> delta_elastic_stress_strain_tensor = elastic_stress_strain_tensor - old_elastic_stress_strain_tensor;
					  double delta_temperature = end_temp - start_temp,
							  	  delta_total_temperature = cell_temp_values[q_point] - old_cell_temp_values[q_point];

					  //declare related parameters at middle time of start time and end time -- tao(j+1)
					  SymmetricTensor<2, dim> tmp_strain_qpoint,
																   tmp_thermal_strain_qpoint,
																   delta_total_strain_qpoint = strain_tensors_i_t_delta_t[q_point] - old_strain_tensors_t[q_point],
																   strain_t = old_strain_tensors_t[q_point] +
																   	   	   	   	   (start_temp - old_cell_temp_values[q_point])/delta_total_temperature*delta_total_strain_qpoint,
																   strain_t_plus_1 = old_strain_tensors_t[q_point] +
																				   (end_temp - old_cell_temp_values[q_point])/delta_total_temperature*delta_total_strain_qpoint,
																   delta_strain_qpoint = delta_temperature/delta_total_temperature*delta_total_strain_qpoint;
					  SymmetricTensor<4, dim>  tmp_elastic_stress_strain_tensor;
					  double tmp_temperature, tmp_miu, tmp_K, tmp_H, tmp_sigma_y0;//, tmp_derivative_sigma_y0, tmp_derivative_H;

					  if (j_iter == 1)
					  { //initialize old_stress, old_plastic_strain, and old_effective_plastic_strain
						  local_quadrature_points_history[q_point].old_stress = local_quadrature_points_history[q_point].pre_stress;//old_elastic_stress_strain_tensor*(local_quadrature_points_history[q_point].old_strain
						  local_quadrature_points_history[q_point].old_plastic_strain = local_quadrature_points_history[q_point].pre_plastic_strain;
						  local_quadrature_points_history[q_point].old_effective_plastic_strain = local_quadrature_points_history[q_point].pre_effective_plastic_strain;
					  }

					  unsigned int q = std::ceil(std::fabs(delta_temperature)/20.0); // number of substeps for each segment
					  if (q == 0 || i == 0) // the first interation of each  load step
						  q = 1;
					  double time_step_seg = fabs(delta_temperature)/fabs(delta_total_temperature)*time_step;
					  substep_size = time_step_seg*1./q;
					  bool bypasscheck = false; // if i ==0, bypass check for stress convergence and start for next integration point
//					  unsigned int j_iter = 1;
					  double time_j = 0;
					  for (unsigned int j = 1; j <= q; j++, j_iter++)
					  {
						  if (i ==0 && j_iter == 1) // the initial displacement loop and first substep for each point, loop k only once
							  substep_size = time_step;

						  // set parameters at middle time of start time and end time: Tao(j+1)_e, Tao(j+1)_e_TH, Tao(j+1)_miu, ...
						  tmp_strain_qpoint = strain_t + delta_strain_qpoint*j*substep_size/time_step_seg;
						  tmp_thermal_strain_qpoint = old_thermal_strain_qpoint + delta_thermal_strain_qpoint*j*substep_size/time_step_seg;
						  tmp_temperature = start_temp + delta_temperature*j*substep_size/time_step_seg;
						  tmp_elastic_stress_strain_tensor = old_elastic_stress_strain_tensor + delta_elastic_stress_strain_tensor*j*substep_size/time_step_seg;

						  SymmetricTensor<2, dim> tmp_plastic_strain_qpoint, pre_plastic_strain,	// tao_j+1_ep(i, k)
																	  stress_j1_k_plus_1, pre_stress;

						  if ((cell_temp_values[q_point] >= melt_point || tmp_temperature >= melt_point) && i >0) // ignore the first elastic loop
						  {
							  time_j += substep_size;
							  local_quadrature_points_history[q_point].old_effective_plastic_strain = 0;
							  local_quadrature_points_history[q_point].old_plastic_strain = 0;
							  if(cell_temp_values[q_point] >= melt_point)
							  {// temperature from Low -> High (melt point)
								  local_quadrature_points_history[q_point].old_stress = elastic_stress_strain_tensor * (strain_t_plus_1 - thermal_strain_qpoint);
								  local_quadrature_points_history[q_point].old_strain = strain_t_plus_1; //strain_tensors_i_t_delta_t[q_point];
								  break;
							  }
							  else
							  {// temperature from High (melt point) -> Low
								  local_quadrature_points_history[q_point].old_stress = tmp_elastic_stress_strain_tensor * (tmp_strain_qpoint - tmp_thermal_strain_qpoint);
								  local_quadrature_points_history[q_point].old_strain = tmp_strain_qpoint;

//								  if ((cell->index() == 0 && q_point == 5))// || fabs(cell_temp_values[q_point] - 1721.98) <1e-2)
//								  {
//									  pcout<<"    ## substep: "<<j_iter
//											  <<", von mises of old stress: "<<get_von_Mises_stress(local_quadrature_points_history[q_point].old_stress)
//											  <<", old strain.norm: "<<tmp_strain_qpoint.norm()<<", delta strain.norm: "<<(tmp_strain_qpoint - tmp_thermal_strain_qpoint).norm()
//											  <<", point temp: "<<tmp_temperature
//											  <<", cell temp: "<<old_cell_temp_values[q_point]<<"->"<<cell_temp_values[q_point]
//											 <<", cell type: "<<old_cell_material[cnt_cells]<<"->"<<cell_material[cnt_cells]
//		//									<<", cell id: "<<cell->index()<<", q: "<<q_point
//											  <<std::endl;
//								  }
								  continue;
							  }
						  }

						  // set other related parameters at middle time of start time and end time: Tao(j+1)_miu, Tao(j+1)_sigma_y0, ...
						  tmp_miu = old_miu + (miu - old_miu)*j*substep_size/time_step_seg;
						  tmp_K = old_K + (K - old_K)*j*substep_size/time_step_seg;
						  tmp_H = old_H + (H - old_H)*j*substep_size/time_step_seg;
						  tmp_sigma_y0 = old_sigma_y0 + (sigma_y0 - old_sigma_y0)*j*substep_size/time_step_seg;
//						  tmp_derivative_sigma_y0 = old_derivative_sigma_y0 + (derivative_sigma_y0 - old_derivative_sigma_y0)*j*substep_size/time_step_seg;
//						  tmp_derivative_H = old_derivative_H + (derivative_H - old_derivative_H)*j*substep_size/time_step_seg;

//						  double flow_constant_qpoint_alpha;//, flow_constant_qpoint_pre;
						  unsigned int max_point_loop_iter = 200;
//						  double tol = 1e-7;
						  unsigned int k_iter = 1;//, interupt_mid = 0;
//						  bool not_yield_flag = false;

						  for (unsigned int k = 0; k < max_point_loop_iter; k++, k_iter++)
						  {// calculate the plastic strain and total stress by using predictor-corrector method
//							  not_yield_flag = false;
//							  flow_constant_qpoint_alpha = 0;

							  tmp_plastic_strain_qpoint = local_quadrature_points_history[q_point].old_plastic_strain;
							  stress_j1_k_plus_1 = tmp_elastic_stress_strain_tensor * (tmp_strain_qpoint - tmp_plastic_strain_qpoint - tmp_thermal_strain_qpoint);

							  if (i == 0)// && timestep_number == 0)
							  {
								  bypasscheck = true;
								  break;
							  }

							  double effective_plastic_strain_j1_k1 = local_quadrature_points_history[q_point].old_effective_plastic_strain;
							  double yield_stress_qpoint_j_plus_1 = tmp_sigma_y0 + tmp_H*effective_plastic_strain_j1_k1;

//							  if (cell->index() == 0 && q_point == 5)
//							  {
//								  pcout<<"    ## substep: "<<j_iter<<", iteration - : "<<k<<", flow const: "<<flow_constant_qpoint_alpha<<", old strain.norm: "<<tmp_strain_qpoint.norm()//<<", delta strain.norm: "<<(tmp_strain_qpoint - tmp_thermal_strain_qpoint).norm()
//	//									  <<", delta plastic strain.norm() = "<<(tmp_plastic_strain_qpoint - local_quadrature_points_history[q_point].old_plastic_strain).norm()
//	//									  <<", von mises of old stress: "<<get_von_Mises_stress(local_quadrature_points_history[q_point].old_stress)
//										  <<", Von Mises stress = "<<get_von_Mises_stress(stress_j1_k_plus_1)
//										  <<", yield stress_j1: "<<yield_stress_qpoint_j_plus_1
//										  <<", effective pla-strain: "<<local_quadrature_points_history[q_point].old_effective_plastic_strain
//										  <<", point temp: "<<tmp_temperature
//										  <<", cell temp: "<<old_cell_temp_values[q_point]<<"->"<<cell_temp_values[q_point]
//										 <<", cell type: "<<old_cell_material[cnt_cells]<<"->"<<cell_material[cnt_cells]
//	//									<<", cell id: "<<cell->index()<<", q: "<<q_point
//										  <<std::endl;
//							  }

							  if (get_von_Mises_stress(stress_j1_k_plus_1) <= yield_stress_qpoint_j_plus_1)
							  {
//								  not_yield_flag = true;
								  break;
							  }

							  // radial return to correct sigma_tao(j+1) and plastic_strain_tao(j+1)
							  double r_delta_t = (deviator(stress_j1_k_plus_1).norm() - sqrt(2./3)*(tmp_sigma_y0 + tmp_H*local_quadrature_points_history[q_point].old_effective_plastic_strain))
																/ (2*tmp_miu + 2./3*tmp_H);
							  double yield_stress_n1 = tmp_sigma_y0 + tmp_H*(local_quadrature_points_history[q_point].old_effective_plastic_strain + sqrt(2./3)*r_delta_t);
							  SymmetricTensor<2, dim> dev_stress_j1_k_plus_1 = sqrt(2./3)*yield_stress_n1*deviator(stress_j1_k_plus_1)/deviator(stress_j1_k_plus_1).norm();

//							  double partial_yield_partial_temperature = (local_quadrature_points_history[q_point].old_effective_plastic_strain + sqrt(2./3)*r_delta_t)*tmp_derivative_H + tmp_derivative_sigma_y0;
//							  double flow_constant_qpoint_test = (tmp_miu*dev_stress_j1_k_plus_1*(delta_total_strain_qpoint - delta_thermal_strain_qpoint)/time_step + dot_miu*2./3*yield_stress_n1*yield_stress_n1/*s_jk_alpha*s_jk_alpha*//(2*tmp_miu)
//																						  - yield_stress_n1/3.*partial_yield_partial_temperature*delta_temperature/time_step)
//																			  /(2./3*yield_stress_n1*yield_stress_n1*(tmp_miu + tmp_H/3.));

							  tmp_plastic_strain_qpoint = local_quadrature_points_history[q_point].old_plastic_strain
																				+ r_delta_t * deviator(stress_j1_k_plus_1)/deviator(stress_j1_k_plus_1).norm();
//							  SymmetricTensor<2, dim> stress_j1_k_plus_1_temp = tmp_elastic_stress_strain_tensor * (tmp_strain_qpoint - tmp_plastic_strain_qpoint - tmp_thermal_strain_qpoint);
							  stress_j1_k_plus_1 = dev_stress_j1_k_plus_1 + tmp_K*trace(tmp_strain_qpoint - tmp_plastic_strain_qpoint - tmp_thermal_strain_qpoint)
																			  *unit_symmetric_tensor<dim>();

//							  if ((cell->index() == 0 && q_point == 5))// || fabs(cell_temp_values[q_point] - 1721.98) <1e-2)
//							  {
//								  pcout<<"                     r_delta_t: "<<r_delta_t<<", yield_stress_n1: "<<yield_stress_n1
//										  <<", after new Von Mises stress = "<<sqrt(3./2)*dev_stress_j1_k_plus_1.norm()
//										  <<", flow const test: "<<flow_constant_qpoint_test
//										  <<", new Von Mises stress = "<<get_von_Mises_stress(stress_j1_k_plus_1)
//										  <<", test Von Mises stress = "<<get_von_Mises_stress(stress_j1_k_plus_1_temp)
//										  <<", effective p-strain: "<<local_quadrature_points_history[q_point].old_effective_plastic_strain
//										  <<std::endl;
//							  }
							  break;
						  }		// end of k --- loop

						  local_quadrature_points_history[q_point].old_effective_plastic_strain +=
												  sqrt(2./3)*(tmp_plastic_strain_qpoint - local_quadrature_points_history[q_point].old_plastic_strain).norm();
						  local_quadrature_points_history[q_point].old_plastic_strain = tmp_plastic_strain_qpoint;
						  local_quadrature_points_history[q_point].old_stress = stress_j1_k_plus_1;
						  local_quadrature_points_history[q_point].old_strain = tmp_strain_qpoint;

						  if (i == 0 && bypasscheck)
							  break;

						  time_j += substep_size;
					  } // end of j --- loop

					  start_temp = end_temp;
					  iter++;
				  }
			  }
		  }
		  }
	  }
  }

  template <int dim>
  void
  HeatEquation<dim>::assemble_newton_system (/*const Vector<double> &incremental_displacement_du, */unsigned int newton_step)
 {
	  TimerOutput::Scope t(computing_timer, "Assembling-mech");

	  hp::FEValues<dim> hp_fe_values(fe_collection_disp, quadrature_collection_disp,
			  	  	  	  	  update_values | update_gradients |
							  update_quadrature_points | update_JxW_values);
	  hp::FEValues<dim> hp_fe_temp_values (fe_collection, quadrature_collection,
			  	  	  	  	  update_values   | update_gradients |
							  update_quadrature_points | update_JxW_values);

	    std::vector<double> cell_temp_values(quadrature_collection[0].size()), old_cell_temp_values(quadrature_collection[0].size());

	  const unsigned int dofs_per_active_cell   = fe_collection_disp[0].dofs_per_cell;
	  const unsigned int n_q_points      = quadrature_collection_disp[0].size();

	  FullMatrix<double>                  cell_matrix(dofs_per_active_cell, dofs_per_active_cell);
	  Vector<double>                         cell_rhs(dofs_per_active_cell);

	  std::vector<types::global_dof_index>   local_dof_indices(dofs_per_active_cell);

//	  std::vector<SymmetricTensor<2, dim> > incremental_strain_tensor(n_q_points);

//	  std::cout<<"incremental_displacement_du = "<<incremental_displacement_du<<std::endl;
	  if (newton_step ==0)
	  {
		  newton_matrix_disp = 0;
		  newton_rhs_disp = 0;
	  }
	  newton_rhs_disp = 0;

	  typename hp::DoFHandler<dim>::active_cell_iterator
	  	  	  	  	  	  	  	  	  cell = dof_handler_disp.begin_active(),
									  endc = dof_handler_disp.end();
	  typename hp::DoFHandler<dim>::active_cell_iterator cell_heat = dof_handler.begin_active();

	  unsigned int cnt_cells (0), active_cnt_cells(0);
	  const FEValuesExtractors::Vector displacement(0);
	  for (; cell != endc; ++cell, ++cell_heat, ++cnt_cells)
	  {
		  if (cell->is_locally_owned())
		  {
		  unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
//		  if (cell->is_locally_owned())
		  if (dofs_per_cell != 0)
		  {
			  active_cnt_cells++;
			  hp_fe_values.reinit(cell);
			  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

			  hp_fe_temp_values.reinit (cell_heat);
			  const FEValues<dim> &fe_temp_values = hp_fe_temp_values.get_present_fe_values();
			  fe_temp_values.get_function_values(solution, cell_temp_values);
//			  fe_temp_values.get_function_values(old_solution, old_cell_temp_values);

			  cell_matrix = 0;
			  cell_rhs = 0;

			  const PointHistory<dim> *local_quadrature_points_history = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
			  Assert (local_quadrature_points_history >= &quadrature_point_history.front(),
					  ExcInternalError());
			  Assert (local_quadrature_points_history < &quadrature_point_history.back(),
					  ExcInternalError());

			  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			  {
				  constitutive_law.set_current_temperature(cell_temp_values[q_point], cell_material[cnt_cells]);
				  SymmetricTensor<4, dim> elastic_stress_strain_tensor;
				  constitutive_law.get_elastic_stress_strain_tensor(elastic_stress_strain_tensor);

				  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				  {
					  if (newton_step == 0)
					  {
						  for (unsigned int j = 0; j < dofs_per_cell; ++j)
							  cell_matrix(i, j) += (fe_values[displacement].symmetric_gradient(i, q_point)*elastic_stress_strain_tensor
									  	  	  	  	  	  	  	  *fe_values[displacement].symmetric_gradient(j, q_point)) * fe_values.JxW(q_point);
					  }
					  cell_rhs(i) += -(local_quadrature_points_history[q_point].old_stress* fe_values[displacement].symmetric_gradient(i, q_point)
						  	  	  	  	  	  	  ) * fe_values.JxW(q_point);
				  }

//				  if (newton_step == 4 && local_quadrature_points_history[q_point].old_stress.norm() > 1e9)
//				  cout<<"cell_rhs: "<<cell_rhs<<" old stress: "<< local_quadrature_points_history[q_point].old_stress<<", cell temperature qpoint: "<<cell_temp_values[q_point]<<std::endl;
			  }

			  cell->get_dof_indices(local_dof_indices);
			  constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_matrix, cell_rhs,
					  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  local_dof_indices,
																							  newton_matrix_disp,
																							  newton_rhs_disp,
																							  true);
		  }
		  }
	  }
//	  std::cout <<"number of active cells: "<<active_cnt_cells<<std::endl;

//	  constraints_dirichlet_and_hanging_nodes.condense(newton_matrix_disp);
//	  constraints_dirichlet_and_hanging_nodes.condense(newton_rhs_disp);

//	  cout<<"before compress of newton_matrix_disp.frobenius_norm="<<newton_matrix_disp.frobenius_norm()<<", rhs.l2_norm="<<newton_rhs_disp.l2_norm()<<", in processor: "<<this_mpi_process<<endl;
	  newton_matrix_disp.compress(VectorOperation::add);
	  newton_rhs_disp.compress(VectorOperation::add);
//	  cout<<"after compress of newton_matrix_disp.frobenius_norm="<<newton_matrix_disp.frobenius_norm()<<", rhs.l2_norm="<<newton_rhs_disp.l2_norm()<<", in processor: "<<this_mpi_process<<endl;
 }

//  template <int dim>
//  void
//  HeatEquation<dim>::solve_newton_system ()
//  {//PETSC doesn't work for mechanical analysis when turning around to another scan track
//	  TimerOutput::Scope t(computing_timer, "Solve-mech");
//
////	  Vector<double> distributed_solution(dof_handler_disp.n_dofs());
//	  LA::MPI::Vector distributed_solution(locally_owned_dofs_disp, mpi_communicator);
//	  distributed_solution = incremental_displacement;		// delta(Uk+1, n)
//
////	  constraints_disp.set_zero(distributed_solution);
////	  constraints_disp.set_zero(newton_rhs_disp);
//
////	  PETScWrappers::PreconditionSSOR  preconditioner;
////	  {
//////		  TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");
////		  PreconditionSSOR<>::AdditionalData additional_data;
////		  preconditioner.initialize(newton_matrix_disp, 1.1);//additional_data);
////	  }
//
//	  {
////		  TimerOutput::Scope t(computing_timer, "Solve: iterate");
////		  Vector<double> tmp(dof_handler_disp.n_dofs());
////		  LA::MPI::Vector tmp(locally_owned_dofs_disp, mpi_communicator);
////		  const double relative_accuracy = 1e-2;
//
//		   double solver_tolerance;//  = relative_accuracy
////				  	  	  	  	  	  	  	  	  	  	  	  * newton_matrix_disp.residual(tmp, distributed_solution, newton_rhs_disp);
//		  	  	  	  	   //newton_rhs_disp - newton_matrix_disp*distributed_solution -->tmp, tmp.l2_norm() is returned
////		  std::cout << "			solver_tolerance = " <<solver_tolerance<<std::endl;
////		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
////		  LA::MPI::Vector tmptt(locally_owned_dofs_disp, mpi_communicator);
////		  newton_matrix_disp.vmult(tmptt,distributed_solution);
////		  pcout << "			newton_matrix_disp*distributed_solution = " <<tmptt.l2_norm()<<std::endl;
//
//		  solver_tolerance = 1e-12;
//
//		  SolverControl solver_control(10*dof_handler_disp.n_dofs(), solver_tolerance); //(10*newton_matrix_disp.m(), solver_tolerance);
//		  PETScWrappers::SolverCG solver (solver_control,
//		  	    	                                mpi_communicator);
//		  PETScWrappers::PreconditionBlockJacobi preconditioner(newton_matrix_disp);
//		  PETScWrappers::PreconditionBlockJacobi::AdditionalData additional_data;
//		  preconditioner.initialize(newton_matrix_disp, additional_data);
////		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
////		  try{
//		  solver.solve(newton_matrix_disp, distributed_solution,
//				  	  	  	  	  	  	  	  	  newton_rhs_disp, preconditioner);	// newton_matrix * distributed_solution = newton_rhs
////		  }
////		  catch (...)
////		  {
////			  std::cerr << std::endl << std::endl
////		                << "----------------------------------------------------"
////		                << std::endl;
////			  std::cerr << "Unknown exception!" << std::endl << "Aborting!"
////		                << std::endl
////		                << "----------------------------------------------------"
////		                << std::endl;
////			  exit(-99);
//////				return 1;
////		  }
//
//		  pcout << "             Error: " << solver_control.initial_value()
//            				<< " -> " << solver_control.last_value() << " in "
//							<< solver_control.last_step() << " CG iterations." //<<"  Max steps = " <<solver_control.max_steps() <<", "<<10*dof_handler_disp.n_dofs()
//							<< std::endl;
//	  }
//	  constraints_dirichlet_and_hanging_nodes.distribute(distributed_solution);
//	  incremental_displacement = distributed_solution;
//  }


//  template <int dim>
//  void
//  HeatEquation<dim>::solve_newton_system ()
//  {//trilinos works for mechanical analysis, but the preconditioner of SSOR makes the solver very slow
//	  TimerOutput::Scope t(computing_timer, "Solve-mech");
//
////	  Vector<double> distributed_solution(dof_handler_disp.n_dofs());
//	  TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs_disp, mpi_communicator);
//	  distributed_solution = incremental_displacement;		// delta(Uk+1, n)
//
//	  constraints_disp.set_zero(distributed_solution);
//	  constraints_disp.set_zero(newton_rhs_disp);
//
//	  TrilinosWrappers::PreconditionSSOR preconditioner;
//	  {
////	      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");
//
//		  TrilinosWrappers::PreconditionSSOR::AdditionalData additional_data;
//		  preconditioner.initialize(newton_matrix_disp, additional_data);
//	  }
//
//	  {
////		  TimerOutput::Scope t(computing_timer, "Solve: iterate");
////		  Vector<double> tmp(dof_handler_disp.n_dofs());
//		  LA::MPI::Vector tmp(locally_owned_dofs_disp, mpi_communicator);
//		  const double relative_accuracy = 0.5e-1;
//
//		   double solver_tolerance = relative_accuracy
//				  	  	  	  	  	  	  	  	  	  	  	  * newton_matrix_disp.residual(tmp, distributed_solution, newton_rhs_disp);
//		  	  	  	  	   //newton_rhs_disp - newton_matrix_disp*distributed_solution -->tmp, tmp.l2_norm() is returned
////		  pcout << "			solver_tolerance = " <<solver_tolerance*100<<std::endl;
////		  cout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
////		  LA::MPI::Vector tmptt(locally_owned_dofs_disp, mpi_communicator);
////		  newton_matrix_disp.vmult(tmptt,distributed_solution);
////		  pcout << "			newton_matrix_disp*distributed_solution = " <<tmptt.l2_norm()<<std::endl;
//
////		  solver_tolerance = 1e-12;
//
//		  SolverControl solver_control(10*newton_matrix_disp.m(), solver_tolerance); //(10*newton_matrix_disp.m(), solver_tolerance);
//		  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
////		  TrilinosWrappers::PreconditionBlockJacobi preconditioner(newton_matrix_disp);
////		  TrilinosWrappers::PreconditionBlockJacobi::AdditionalData additional_data;
////		  preconditioner.initialize(newton_matrix_disp, additional_data);
////		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
//		  solver.solve(newton_matrix_disp, distributed_solution,
//				  	  	  	  	  	  	  	  	  newton_rhs_disp, preconditioner);	// newton_matrix * distributed_solution = newton_rhs
//
//		  pcout << "             Error: " << solver_control.initial_value()
//            				<< " -> " << solver_control.last_value() << " in "
//							<< solver_control.last_step() << " CG iterations."// <<"  Max steps = " <<solver_control.max_steps() <<", "<<10*dof_handler_disp.n_dofs()
//							<< std::endl;
//	  }
//	  constraints_dirichlet_and_hanging_nodes.distribute(distributed_solution);
//	  incremental_displacement = distributed_solution;
//  }


//  template <int dim>
//  void
//  HeatEquation<dim>::solve_newton_system ()
//  { //trilinos works for mechanical analysis, preconditioner of PreconditionBlockJacobi makes the solver faster than SSOR
//	  TimerOutput::Scope t(computing_timer, "Solve-mech");
//
////	  Vector<double> distributed_solution(dof_handler_disp.n_dofs());
//	  LA::MPI::Vector distributed_solution(locally_owned_dofs_disp, mpi_communicator);
//	  distributed_solution = incremental_displacement;		// delta(Uk+1, n)
//
////	  constraints_disp.set_zero(distributed_solution);
////	  constraints_disp.set_zero(newton_rhs_disp);
////	  constraints_dirichlet_and_hanging_nodes.set_zero(distributed_solution);
////	  constraints_dirichlet_and_hanging_nodes.set_zero(newton_rhs_disp);
//
//
////	  TrilinosWrappers::PreconditionSSOR preconditioner;
////	  {
//////	      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");
////
////		  TrilinosWrappers::PreconditionSSOR::AdditionalData additional_data;
////		  preconditioner.initialize(newton_matrix_disp, additional_data);
////	  }
//
//	  {
////		  TimerOutput::Scope t(computing_timer, "Solve: iterate");
////		  Vector<double> tmp(dof_handler_disp.n_dofs());
////		  LA::MPI::Vector tmp(locally_owned_dofs_disp, mpi_communicator);
////		  const double relative_accuracy = 1e-2;
//
//		   double solver_tolerance;//  = relative_accuracy
////				  	  	  	  	  	  	  	  	  	  	  	  * newton_matrix_disp.residual(tmp, distributed_solution, newton_rhs_disp);
//		  	  	  	  	   //newton_rhs_disp - newton_matrix_disp*distributed_solution -->tmp, tmp.l2_norm() is returned
////		  std::cout << "			solver_tolerance = " <<solver_tolerance<<std::endl;
////		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
////		  LA::MPI::Vector tmptt(locally_owned_dofs_disp, mpi_communicator);
////		  newton_matrix_disp.vmult(tmptt,distributed_solution);
////		  pcout << "			newton_matrix_disp*distributed_solution = " <<tmptt.l2_norm()<<std::endl;
//
//		  solver_tolerance = 1e-12;
//
//		  SolverControl solver_control(10*dof_handler_disp.n_dofs(), solver_tolerance); //(10*newton_matrix_disp.m(), solver_tolerance);
////		  TrilinosWrappers::SolverCG solver (solver_control,
////		  	    	                                mpi_communicator);
//		  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
//		  TrilinosWrappers::PreconditionBlockJacobi preconditioner;//(newton_matrix_disp);
//		  TrilinosWrappers::PreconditionBlockJacobi::AdditionalData additional_data;
//		  preconditioner.initialize(newton_matrix_disp, additional_data);
////		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
//		  solver.solve(newton_matrix_disp, distributed_solution,
//				  	  	  	  	  	  	  	  	  newton_rhs_disp, preconditioner);	// newton_matrix * distributed_solution = newton_rhs
//
//		  pcout << "             Error: " << solver_control.initial_value()
//            				<< " -> " << solver_control.last_value() << " in "
//							<< solver_control.last_step() << " CG iterations."// <<"  Max steps = " <<solver_control.max_steps() <<", "<<10*dof_handler_disp.n_dofs()
//							<< std::endl;
//	  }
//	  constraints_dirichlet_and_hanging_nodes.distribute(distributed_solution);
//	  incremental_displacement = distributed_solution;
//  }

  TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
  template <int dim>
  void
  HeatEquation<dim>::solve_newton_system ()
  { //trilinos works for mechanical analysis, preconditioner of AMG makes the solver faster than SSOR but lower than PreconditionBlockJacobi
	 //SolverBicgstab with preconditioner of AMG is faster
	  TimerOutput::Scope t(computing_timer, "Solve-mech");

//	  Vector<double> distributed_solution(dof_handler_disp.n_dofs());
	  LA::MPI::Vector distributed_solution(locally_owned_dofs_disp, mpi_communicator);
	  distributed_solution = incremental_displacement;		// delta(Uk+1, n)

	  constraints_disp.set_zero(distributed_solution);
	  constraints_disp.set_zero(newton_rhs_disp);

	  TrilinosWrappers::PreconditionAMG preconditioner;
	  {
////	      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");
//
//		  TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
		  if (mesh_changed_flg_for_mech)
		  {
			    mesh_changed_flg_for_mech = false;
		        std::vector<std::vector<bool> > constant_modes;
		        DoFTools::extract_constant_modes(dof_handler_disp, ComponentMask(),
		                                         constant_modes);
		        additional_data.constant_modes = constant_modes;
		        additional_data.elliptic = true;
		        additional_data.n_cycles = 1;
		        additional_data.w_cycle = false;
		        additional_data.output_details = false;
		        additional_data.smoother_sweeps = 2;
		        additional_data.aggregation_threshold = 1e-2;
		  }
		  preconditioner.initialize(newton_matrix_disp, additional_data);
	  }

	  {
//		  TimerOutput::Scope t(computing_timer, "Solve: iterate");
		  LA::MPI::Vector tmp(locally_owned_dofs_disp, mpi_communicator);
		  const double relative_accuracy = 0.5e-1;

		   double solver_tolerance = relative_accuracy
				  	  	  	  	  	  	  	  	  	  	  	  * newton_matrix_disp.residual(tmp, distributed_solution, newton_rhs_disp);
		  	  	  	  	   //newton_rhs_disp - newton_matrix_disp*distributed_solution -->tmp, tmp.l2_norm() is returned
//		  std::cout << "			solver_tolerance = " <<solver_tolerance<<std::endl;
//		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
//		  LA::MPI::Vector tmptt(locally_owned_dofs_disp, mpi_communicator);
//		  newton_matrix_disp.vmult(tmptt,distributed_solution);
//		  pcout << "			newton_matrix_disp*distributed_solution = " <<tmptt.l2_norm()<<std::endl;

//		  solver_tolerance = 1e-12;

		  SolverControl solver_control(10*newton_matrix_disp.m(), solver_tolerance); //(10*newton_matrix_disp.m(), solver_tolerance);
		  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
//		  SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);
//		  TrilinosWrappers::PreconditionBlockJacobi preconditioner;//(newton_matrix_disp);
//		  TrilinosWrappers::PreconditionBlockJacobi::AdditionalData additional_data;
//		  preconditioner.initialize(newton_matrix_disp, additional_data);
//		  pcout << "			newton_rhs_disp = " <<newton_rhs_disp.l2_norm()<<std::endl;
		  solver.solve(newton_matrix_disp, distributed_solution,
				  	  	  	  	  	  	  	  	  newton_rhs_disp, preconditioner);	// newton_matrix * distributed_solution = newton_rhs

		  pcout << "             Error: " << solver_control.initial_value()
            				<< " -> " << solver_control.last_value() << " in "
							<< solver_control.last_step() << " CG iterations.";
//							<< std::endl;
	  }
//	  cout<<"distributed_solution.l2_norm="<<distributed_solution.l2_norm()<<", in processor: "<<this_mpi_process<<endl;

	  incremental_displacement = distributed_solution;
	  constraints_dirichlet_and_hanging_nodes.distribute(incremental_displacement);

  }

//  template <int dim>
//  void
//  HeatEquation<dim>::solve_newton_system ()
//  { //trilinos works for mechanical analysis, direct solver
//	  TimerOutput::Scope t(computing_timer, "Solve-mech");
//
////	  Vector<double> distributed_solution(dof_handler_disp.n_dofs());
//	  LA::MPI::Vector distributed_solution(locally_owned_dofs_disp, mpi_communicator);
//	  distributed_solution = incremental_displacement;		// delta(Uk+1, n)
//
//	  constraints_disp.set_zero(distributed_solution);
//	  constraints_disp.set_zero(newton_rhs_disp);
////	  constraints_dirichlet_and_hanging_nodes.set_zero(distributed_solution);
////	  constraints_dirichlet_and_hanging_nodes.set_zero(newton_rhs_disp);
//
//	  {
//
////		  LA::MPI::SparseMatrix transpose_system_matrix;
////		  transpose_system_matrix.reinit(newton_matrix_disp);
////		  transpose_system_matrix.copy_from(newton_matrix_disp);
////		  transpose_system_matrix.transpose();
////		  transpose_system_matrix.add(-1, newton_matrix_disp);
////		  cout<<"frobenius_norm = "<<transpose_system_matrix.frobenius_norm()<<
////				  ", l1_norm = "<<transpose_system_matrix.l1_norm()<<
////				  ", linfty_norm = "<<transpose_system_matrix.linfty_norm()<<", in processor: "<<this_mpi_process<<endl;
//
////		  std::ofstream out("matrix_printout.txt");
////		  newton_matrix_disp.print(out, true);
//
//		  SolverControl solver_control (1,0);
//		  TrilinosWrappers::SolverDirect::AdditionalData data;
//		  data.solver_type = "Amesos_Superludist";
//		  TrilinosWrappers::SolverDirect direct (solver_control, data);
//		  direct.solve (newton_matrix_disp, distributed_solution, newton_rhs_disp);
//
//		  pcout << "             Error: " << solver_control.initial_value()
//            				<< " -> " << solver_control.last_value() << " in "
//							<< solver_control.last_step() << " CG iterations."// <<"  Max steps = " <<solver_control.max_steps() <<", "<<10*dof_handler_disp.n_dofs()
//							<< std::endl;
//	  }
//	  constraints_dirichlet_and_hanging_nodes.distribute(distributed_solution);
//	  incremental_displacement = distributed_solution;
//  }



  template<int dim>
  void HeatEquation<dim>::mechanical_run(unsigned int current_timestep_number)//, Vector<double> input_solution_temp)
  {
	  pcout<<"       ******start to solve the mechanical equation...timestep--"<< current_timestep_number<<std::endl;

	  pcout << "   --- Number of active cells in Mechanical Dof handler:="
	              << triangulation.n_active_cells()
	              << std::endl;
      pcout << "   --- Number of degrees of freedom in Mechanical Dof handler: ="
                << dof_handler_disp.n_dofs()
                << std::endl;

	  Vector<double> old_solution_disp(dof_handler_disp.n_dofs());
	  Vector<double> old_incremental_displacement(incremental_displacement.size()), delta_incremental_displacement(incremental_displacement.size());
	  pcout << "			solution_disp.l2_norm = " <<solution_disp.l2_norm()<<", processor:"<<this_mpi_process<<std::endl;

	  old_solution_disp = solution_disp;
      double tol = 0.5e-6; // tolerance
	  const unsigned int max_newton_iter = 100;
	  for (unsigned int i = 0; i <= max_newton_iter; ++i)
	  {// displacement loop, i= 0, 1, 2, ...
//		  if (i ==0)
//			  tmp_solution = solution_disp;

		  pcout << "	  ############ iteration: "<< i <<" ###########"<< std::endl;
//		  pcout << "			calculate for integration points --start" << std::endl;
		  calculate_stress_strain_for_integration_points(i, old_solution_disp);
//		  pcout << "			calculate for integration points --end" << std::endl;


//		  pcout << "			assemble --start" << std::endl;
//		  newton_matrix_disp = 0;
//		  newton_rhs_disp = 0;
		  assemble_newton_system(/*incremental_displacement, */i);
//		  pcout << "			assemble --end" << std::endl;
		  solve_newton_system(); 	// K*du = R -> solve for du  (incremental_displacement)

		  solution_disp +=  incremental_displacement; // update solution_disp
//		  cout << "		incremental_displacement.l2_norm():"<<incremental_displacement.l2_norm() <<", processor:"<<this_mpi_process<<std::endl;
		  pcout << ", incremental_displacement.l2_norm() = "<<incremental_displacement.l2_norm() <<std::endl;

		  delta_incremental_displacement = old_incremental_displacement;
		  delta_incremental_displacement -= incremental_displacement;

//		  cout<<"l2_norm = "<<incremental_displacement.l2_norm()<<
//				  ", l1_norm = "<<incremental_displacement.l1_norm()<<
//				  ", linfty_norm = "<<incremental_displacement.linfty_norm()<<", in processor: "<<this_mpi_process<<endl;

		  if (incremental_displacement.l2_norm() < tol || fabs(delta_incremental_displacement.l2_norm()) < tol*1e-3)
			  break;
		  old_incremental_displacement = incremental_displacement;
	  }

	  update_quadrature_point_history(); // update pre_stress and pre_strain

      move_mesh(true);	// distortion
	  output_mech_results();
      move_mesh(false);	// go back to the original mesh
	  pcout<<"       ******end of solving the mechanical equation..." <<std::endl;
  }

  template<int dim>
  void HeatEquation<dim>::run()
  {
	  thermal_mechanical_flg = false; // close the mechanical analysis

	  pcout << "Parsing my XML file..." << std::endl;
	  xml_document<> doc;
	  xml_node<> * root_node;
	  // Read the xml file into a vector
	  std::ifstream theFile (layer_file_name);
	  std::vector<char> buffer((std::istreambuf_iterator<char>(theFile)), std::istreambuf_iterator<char>());
	  buffer.push_back('\0');
	  // Parse the buffer using the xml file parsing library into doc
	  doc.parse<0>(&buffer[0]);
	  // Find the root node
	  root_node = doc.first_node();	// Layers
	  xml_node<> *machine_node = root_node->first_node();	// Machine_node

	  pcout<<"Total_layers = "<<machine_node->first_node("Total_layers")->value()
			   <<", Velocity = "<<machine_node->first_node("Velocity")->value()
			  <<", Beam_diameter = "<<machine_node->first_node("Beam_diameter")->value()<<endl<<endl;

	  total_layers = atoi(machine_node->first_node("Total_layers")->value());
	  velocity = atof(machine_node->first_node("Velocity")->value());
	  beam_diameter = atof(machine_node->first_node("Beam_diameter")->value());
	  max_segment_length = atof(machine_node->first_node("Max_segment_length")->value()); // for line segment
	  min_segment_length = atof(machine_node->first_node("Min_segment_length")->value()); // for line segment
	  max_segment_length = 2.e-3;

	  computing_timer.reset();
	  const int initial_global_refinement = 4;//std::log2(0.05e-3/0.05e-3); // == 4
	  // create the coarse grid
	  create_coarse_grid ();
//	  if (thermal_mechanical_flg) // initialize stress and strain at quadrature points
	  setup_quadrature_point_history ();

//start_time_iteration:
	{
		// get the first layer node
		xml_node<> *layer_id_node = machine_node->next_sibling();
		xml_node<> *layer_node = layer_id_node->first_node();	//Layer_node
		get_attributes_in_layer_node(layer_node);
	}

	// Set the right FE type for each cell
	set_active_fe_indices();
	// Initialize the matrices and RHS. Setup the dof_handler, boundaries, FE_type and cell_material
	setup_system();
	old_solution.reinit(dof_handler.n_dofs());

	// INITIAL CONDITION
	LA::MPI::Vector local_solution;
	local_solution.reinit(locally_owned_dofs, mpi_communicator);
	EquationData::InitialCondition<dim> initial_condition(Tinit);  //Tinit
	VectorTools::interpolate(dof_handler,
    							initial_condition,
								local_solution);
	constraints.distribute(local_solution);
	old_solution = local_solution;
	solution = old_solution;

	output_results();//return;

	int loop_cnt = 0;
	for (xml_node<> *layer_id_node = machine_node->next_sibling(); layer_id_node; layer_id_node = layer_id_node->next_sibling())
	{	// iterate over layers
		xml_node<> *layer_node = layer_id_node->first_node();	//Layer_node
		pcout<<endl<<"Layer_id = "<<layer_node->first_node("Layer_id")->value()<<
					", Total_scan_tracks = "<<layer_node->first_node("Total_scan_tracks")->value()<<
					", Orientation = "<<layer_node->first_node("Orientation")->value()<<
					", Thickness = "<<layer_node->first_node("Thickness")->value()<<
					", Hatching_space = "<<layer_node->first_node("Hatching_space")->value()<<
					", Part_height = "<<layer_node->first_node("Part_height")->value()<<
					", Layer_start_point = "<<layer_node->first_node("Layer_start_point")->value()<<
					", Layer_end_point = "<<layer_node->first_node("Layer_end_point")->value()<<
					", Layer_start_time = "<<layer_node->first_node("Layer_start_time")->value()<<
					", Layer_end_time = "<<layer_node->first_node("Layer_end_time")->value()<<
					", Idle_time = "<<layer_node->first_node("Idle_time")->value()<<endl;

		theta = 0.5;
		get_attributes_in_layer_node(layer_node);
		for (xml_node<> *scan_track_id = layer_node->next_sibling(); scan_track_id; scan_track_id = scan_track_id->next_sibling())
		{	// iterate over scanning tracks
			xml_node<> *track_node = scan_track_id->first_node();	//Track_node
			pcout<<endl<<"-Track_id = "<<track_node->first_node("Track_id")->value()<<
						", Scan_velocity = "<<track_node->first_node("Scan_velocity")->value()<<
						", Time_step_point = "<<track_node->first_node("Time_step_point")->value()<<
						", Time_step_line = "<<track_node->first_node("Time_step_line")->value()<<endl;

	  		get_attributes_in_track_node(track_node);
	  		unsigned int cnt_segment = 1;
	  		for (xml_node<> *segment_id = track_node->next_sibling(); segment_id; segment_id = segment_id->next_sibling(), cnt_segment++)
	  		{	// iterate over segments
	  			get_attributes_in_segment_node(segment_id);
	  			pcout<<endl<<"--Segment_"<<cnt_segment<<
	  									", Source_type = "<<source_type.c_str()<<
										", Segment_length = "<<segment_id->first_node("Segment_length")->value()<<
										", Segment_start_time = "<<segment_id->first_node("Segment_start_time")->value()<<
										", Segment_end_time = "<<segment_id->first_node("Segment_end_time")->value()<<
										", Segment_start_point = "<<segment_id->first_node("Segment_start_point")->value()<<
										", Segment_end_point = "<<segment_id->first_node("Segment_end_point")->value()<<endl;

	  			// temperary added - 09/23
//	  			if(segment_length < 0.00051)
//	  				continue;
				{
	  				if (timestep_number == 0)// && thermal_mechanical_flg)
	  					setup_mech_system();
	  				unsigned int min_level = 4, max_level = 5;
//	  				determine_refine_level(layer_id, min_level, max_level);
	  				if(cnt_segment < 2)// && track_id == 1)		//(segment_length > 0.00051)
	  				{
//	  					if (timestep_number > 1){
//	  						timestep_number +=1000;
//	  						move_mesh(true);	// distortion
//	  						output_mech_results();
//	  						move_mesh(false);	// go back to the original mesh
//	  						timestep_number -= 1000;
//	  					}
//	  					refine_mesh (min_level, max_level);
//	  					if (timestep_number > 1){
//	  						timestep_number +=1001;
//	  						move_mesh(true);	// distortion
//	  						output_mech_results();
//	  						move_mesh(false);	// go back to the original mesh
//	  						timestep_number -= 1001;
//	  					}
	  				}
//					if (cell_material.size() != old_cell_material.size())
//						old_cell_material = cell_material;
//					if(cnt_segment < 2)	//(segment_length > 0.00051)
//						setup_system();
				}

//	  			double middle_pre_refinement_step = 0;
				while (time < segment_end_time)
				{
					if (source_type == "Point")
						time_step = time_step_point;
					else // source_type == "Line"
	  					time_step = time_step_line;

					time += time_step;
					++timestep_number;
					if (time > segment_end_time)
					{
						time_step -= (time - segment_end_time);
						time = segment_end_time;
						if (source_type == "Line")
						{// the last time step for line heat source may be wrong. here we are trying to correct the surface thermal distribution
//							time_step_line = segment_length/scan_velocity/(num_dt_line+1);
							segment_start_time += time_step_line;
						}
					}

//					set_active_fe_indices();
//					setup_system();

//					std::cout << std::endl << "***Time step " << timestep_number << " at t=" << time
//							  << std::endl;
//					std::cout << "===========================================" << std::endl
//				                    << "Number of active cells: " << triangulation.n_active_cells()  << std::endl
//									<< "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;
//					solve_relaxed_Picard();

					output_results();
					{// mechanical analysis
//						setup_mech_system();
//						mechanical_run(timestep_number);//, solution);
					}
//					store_old_vectors_disp();

					old_solution = solution;
					old_cell_material = cell_material;

					computing_timer.print_summary();
					computing_timer.reset();
				}

	  		}
		}

// heat transfer for the idle time after each layer
//    	time_step = 0.1
// here, time = layer_end_time.
//
  		pcout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! COOLING FOR 1 SECOND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;

  		double pre_time_step = 0.003/*0.0003time_step*/, cnt_cooling_step = 0;  //pre_time_step = 0.000333

  		theta = 0.9;
  		int timestep_number_in_cooling = -1;

  		if(layer_id == 30)
  			idle_time = 350;
  		while(time < layer_end_time + idle_time)
  		{

  			time_step = 0.005*pre_time_step*pow(cnt_cooling_step+1, 3);//0.005*pre_time_step*pow(cnt_cooling_step+1, 3);

  			if(time > layer_end_time+10)
  				time_step = 5;
  			cnt_cooling_step++;

  			time += time_step;	// layer_end_time < time <= idle time
  			++timestep_number;	timestep_number_in_cooling++;
			if (time > layer_end_time + idle_time)
			{
				time_step -= (time - layer_end_time - idle_time);
				time = layer_end_time + idle_time;
			}

//  			std::cout << std::endl << "***Time step " << timestep_number << " at t=" << time
//					  << std::endl;
//  			std::cout << "===========================================" << std::endl
//		                    << "Number of active cells: " << triangulation.n_active_cells()  << std::endl
//							<< "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;
//  			solve_system();

  		    output_results();
//  		    {// mechanical analysis
//  		    	setup_mech_system();
//  		    	mechanical_run(timestep_number);//, solution);
//  		    }

			old_solution = solution;
			old_cell_material = cell_material;
  		}

//  		refine_mesh_btw_layers(part_height, initial_global_refinement);

//		  timestep_number +=998;
//	      move_mesh(true);	// distortion
//		  output_mech_results();
//	      move_mesh(false);	// go back to the original mesh
//	      output_results();
//	      timestep_number -= 998;

  		old_solution = solution; old_cell_material = cell_material;

		pcout<<"entered store old vectors start!!!!!"<<endl;
//		store_old_vectors();
//		store_old_vectors_disp();
		pcout<<"entered store old vectors end!!!!!"<<endl;

		pcout<<"before part height: "<<part_height<<std::endl;
		part_height = (layer_id + 1)*thickness;
		pcout<<"after part height: "<<part_height<<std::endl;

//		set_active_fe_indices();	// activate a new layer material
//		setup_system();	// dof will be changed
//		setup_mech_system();

		pcout<<"entered transfer old vectors start!!!!!"<<endl;
//		transfer_old_vectors();	// transfer_old_vectors immediately after activate a new layer
//		transfer_old_vectors_disp();
		pcout<<"entered transfer old vectors end!!!!!"<<endl;

//		  timestep_number +=999;
//	      move_mesh(true);	// distortion
//		  output_mech_results();
//	      move_mesh(false);	// go back to the original mesh
//	      solution = old_solution;
//	      output_results();
//	      timestep_number -= 999;

  		loop_cnt++;
    }

  }


}


int main(int argc, char *argv[])
{
	try
	{
		using namespace std;
		using namespace dealii;
		using namespace Thermo_Elasto_Plastic_Space;

		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

//		if (argc != 3)
//		{
//			std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
//			std::exit(1);
//		}
		string str1= "parameters.prm";
		string str2= "codebeautify.xml";
		const char * char_1 = str1.c_str();
		const char * char_2 = str2.c_str();

//		HeatEquation<3> heat_equation_solver(argv[1], argv[2]);
		HeatEquation<3> heat_equation_solver(char_1, char_2);
		Timer time_couter;
		time_couter.start();
		heat_equation_solver.run();

//Test for mechanical material property -- start
//		{
//			Parameters::AllParameters<3>  parameters(char_1);
//			ConstitutiveLaw<3> constitutive_law(parameters);
//			MaterialData<3> mat_data(parameters);
//			double E0 = mat_data.get_elastic_modulus(2333, 0);
//			double E1 = mat_data.get_elastic_modulus(2333, 1);
//			double E2 = mat_data.get_elastic_modulus(2333, 2);
//			std::cout <<"elastic_modulus 0= "<<E0 << std::endl;
//			std::cout <<"elastic_modulus 1= "<<E1 << std::endl;
//			std::cout <<"elastic_modulus 2= "<<E2 << std::endl;
//
//			double v0 = mat_data.get_poisson_ratio(2333, 0);
//			double v1 = mat_data.get_poisson_ratio(2333, 1);
//			double v2 = mat_data.get_poisson_ratio(2333, 2);
//			std::cout <<"poisson_ratio 0= "<<v0 << std::endl;
//			std::cout <<"poisson_ratio 1= "<<v1 << std::endl;
//			std::cout <<"poisson_ratio 2= "<<v2 << std::endl;
//
//			double a0 = mat_data.get_thermal_expansion(2333, 0);
//			double a1 = mat_data.get_thermal_expansion(2333, 1);
//			double a2 = mat_data.get_thermal_expansion(2333, 2);
//			std::cout <<"thermal_expansion 0= "<<a0 << std::endl;
//			std::cout <<"thermal_expansion 1= "<<a1 << std::endl;
//			std::cout <<"thermal_expansion 2= "<<a2 << std::endl;
//
//			double y0 = mat_data.get_elastic_limit(2333, 0);
//			double y1 = mat_data.get_elastic_limit(2333, 1);
//			double y2 = mat_data.get_elastic_limit(2333, 2);
//			std::cout <<"yield_strength 0= "<<y0 << std::endl;
//			std::cout <<"yield_strength 1= "<<y1 << std::endl;
//			std::cout <<"yield_strength 2= "<<y2 << std::endl;
//
//			double h0 = mat_data.get_hardening_parameter(2333, 0);
//			double h1 = mat_data.get_hardening_parameter(2333, 1);
//			double h2 = mat_data.get_hardening_parameter(2333, 2);
//			std::cout <<"hardening 0= "<<h0 << std::endl;
//			std::cout <<"hardening 1= "<<h1 << std::endl;
//			std::cout <<"hardening 2= "<<h2 << std::endl;
//		}
//Test for mechanical material property -- start

		double endtime = time_couter.stop();
		double walltime = time_couter.wall_time();
//		double laptime = time_couter.get_lap_time();
		std::cout <<"endtime="<<endtime << std::endl;
		std::cout <<"walltime="<<walltime << std::endl;
//		std::cout <<"laptime="<<laptime << std::endl;

	}

	catch (std::exception &exc)
    {
		std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
		std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

		return 1;
    }
	catch (...)
	{
		std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
		std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
		return 1;
	}

	return 0;
}
