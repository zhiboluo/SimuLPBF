/*
 * EquationData.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: zhiboluo
 */


namespace EquationData
{
	using namespace dealii;


	template<int dim>
	class InitialCondition : public Function<dim>
	{
	public:
		InitialCondition (const double T_init)
		:
			Function<dim>()
			{
			Tinit = T_init;
			}
		virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
		double Tinit;
	};

	template<int dim>
	double InitialCondition<dim>::value (const Point<dim> &/*p*/, const unsigned int component) const
	{
		(void) component;
		Assert (component == 0, ExcInternalError());
		return Tinit; // Initial Temperature value
	}

	template<int dim>
	class RightHandSide : public Function<dim>
	{
	public:
		RightHandSide ()
		:
			Function<dim>()
			{}

		virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
	};

  template<int dim>
  double RightHandSide<dim>::value (const Point<dim> &/*p*/,
                                      const unsigned int component) const
  {
	  (void) component;
      Assert (component == 0, ExcInternalError());
	  return 0;
  }


  //Geometry for 3D model
  // ---- Part
  const double g_width = 1.6e-3;//0.4e-3; //should be integer multiples of 0.05e-3
  const double g_length = 20.8e-3;//2e-3; // should be integer multiples of 0.05e-3
  const double g_height = 1.6e-3; //0.05e-3;
  // ---- Baseplate
  const double g_base_length = 72e-3; //should be integer multiples of 0.05e-3*2^n. Here, n = 11.
  const double g_base_width = 25.6e-3; //should be integer multiples of 0.05e-3*2^n. Here, n = 11.
  const double g_base_height = 6.4e-3; //should be integer multiples of 0.05e-3*2^n. Here, n = 8.

  template<int dim>
  class InputHeatFlux : public Function<dim>
  {
  public:
	  InputHeatFlux (const Parameters::AllParameters<dim> &input_para, const std::string source,
			  const double velocity, const double start_time, const double end_time, const double line_length, const double orientation, const Point<dim> start_point, const Point<dim> end_point)
      :
      Function<dim>(),
	  par(input_para),
      period (par.line_length/par.v),
	  v(par.v),
	  w(par.w),
	  P(par.P),
	  Emsvity(par.Emsvity),
	  source_type(source),
	  scan_velocity(velocity),
	  segment_start_time(start_time),
	  segment_end_time(end_time),
	  segment_length(line_length),
	  segment_start_point(start_point),
	  segment_end_point(end_point),
	  theta(orientation)//*M_PI/180.)
    {
		  double a = w, b = 1*w, c = 1*w;
		  if (source_type == "Point")
			  I0 = 6*sqrt(3)*(1-Emsvity)*P*1.0/(M_PI*sqrt(M_PI)*a*b*c);
		  else if (source_type == "Line")
		  {
			  const double dt = segment_length/scan_velocity;
			  I0 = (1-Emsvity)*P*pow(M_PI/2.0, 0.5)/dt/M_PI/w/scan_velocity;
		  }

		  theta = acos((segment_end_point[0] - segment_start_point[0])
				  /sqrt(pow(segment_end_point[0] - segment_start_point[0], 2) +
						  pow(segment_end_point[1] - segment_start_point[1], 2) +
						  pow(segment_end_point[2] - segment_start_point[2], 2)));
		  if (segment_end_point[1] - segment_start_point[1] < 0)
			  theta = -theta;
//		  if (source_type == "Line")
//		  {
//			  const double time = this->get_time();
//			  double dt = segment_length/scan_velocity;
//			  std::cout<<"theta = "<<theta<<std::endl;
//			  std::cout<<"segment_start_point[0] = "<<segment_start_point[0]<<std::endl;
//			  std::cout<<"segment_start_point[1] = "<<segment_start_point[1]<<std::endl;
//			  std::cout<<"segment_length = "<<segment_length<<std::endl;
//			  std::cout<<"dt = "<<segment_length/scan_velocity<<std::endl;
//			  std::cout<<"time_temp = "<<std::floor((time-segment_start_time)/dt)*dt<<std::endl;
//			  std::cout<<"vx = "<<scan_velocity*cos(theta)<<std::endl;
//			  std::cout<<"vy = "<<scan_velocity*sin(theta)<<std::endl;
//		  }
    }

	  virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
//      virtual void value_list (const std::vector<Point<dim> > &points,
//                         std::vector<double >   &value_list) /*const*/;

  	  private:
	  	  Parameters::AllParameters<dim> par;
	  	const double period;
	  	  // heat source parameters
	  	const double v; //scanning speed = 5 cm/s
	  	const double w;	// radius of laser spot
	  	const double P;//*1000;		// laser power 1 W = 1 Kg*m^2/s^3 = 1000 g*m^2/s^3
	  	const double Emsvity;	//emissivity coefficient of laser
	  	double I0;// = 2*(1-Emsvity)*P/M_PI/std::pow(w, 2);	// maximum laser heat input

	  	  // printing parameters for segment node
	  	  std::string source_type;
	  	  double scan_velocity;
	  	  double segment_start_time, segment_end_time;
	  	  double segment_length;
	  	  Point<dim> segment_start_point;
	  	  Point<dim> segment_end_point;
	  	  double theta;
  };

  template<int dim>
  double InputHeatFlux<dim>::value (const Point<dim> &p,
                                      const unsigned int component) const
  {
	  (void) component;
	  Assert (component == 0, ExcInternalError());

	  const double time = this->get_time();
	  double return_val = 0;

	  //if (dim == 3)
	  if (time <= segment_end_time)
	  {
		  double vx = scan_velocity*cos(theta), vy = scan_velocity*sin(theta);
		  {// scanning direction: left ---> right
			  if (source_type == "Point")
			  { // Point heat source
//				  return_val = I0*exp(-2*(pow(p[1] - vy*(time - segment_start_time) - segment_start_point[1], 2) +
//						  	  	  	  	  	  	  	  	  pow(p[0] - vx*(time - segment_start_time) - segment_start_point[0], 2))/pow(w, 2));
				  // ellipsoidal volumetric heat flux model
				  double a = w, b = 1*w, c = 1*w;
//				  if (p[0] - vx*(time - segment_start_time) - segment_start_point[0] < 0)
//					  a = 1*w;
				  return_val = I0*exp(-3*(pow(p[1] - vy*(time - segment_start_time) - segment_start_point[1], 2)/pow(b, 2) +
						  	  	  	  	  	  	  	  	  pow(p[0] - vx*(time - segment_start_time) - segment_start_point[0], 2)/pow(a, 2) +
														  pow(p[2] - segment_start_point[2], 2)/pow(c, 2)));
			  }
			  else if (source_type == "Line")
			  { // Line heat source
				  const double dt = segment_length/scan_velocity;
				  double time_temp = floor((time-segment_start_time)/dt)*dt;
//				  time_temp = fabs(time_temp);
				  return_val = I0*exp(-2*pow( - (p[0] - vx*time_temp - segment_start_point[0])*sin(theta) +
						  	  	  	  	  	  	  	  (p[1] - vy*time_temp - segment_start_point[1])*cos(theta), 2)/pow(w, 2))*
				  	  	  	  	  	  (- erf(pow(2, 0.5)*((p[0] - vx*(time_temp + dt) - segment_start_point[0])*cos(theta) +
				  	  	  	  	  			  (p[1] - vy*(time_temp + dt) - segment_start_point[1])*sin(theta))/w)
				  	  	  	  	  	   + erf(pow(2, 0.5)*((p[0] - vx*time_temp - segment_start_point[0])*cos(theta) +
							  	  	  	  	  (p[1] - vy*time_temp - segment_start_point[1])*sin(theta))/w));
			  }
		  }

	  }

	  return return_val;
  }

//  template <int dim>
//  void
//  InputHeatFlux<dim>::
//  value_list (const std::vector<Point<dim> > &points,
//                     std::vector<double >   &value_list) //const
//  {
//    const unsigned int n_points = points.size();
//
//    Assert (value_list.size() == n_points,
//            ExcDimensionMismatch (value_list.size(), n_points));
//
//    for (unsigned int p=0; p<n_points; ++p)
//    	value_list[p] = InputHeatFlux<dim>::value (points[p]);
//  }


  template<int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value (const Point<dim>  &p,
                          const unsigned int component = 0) const;
  };

  template<int dim>
  double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                     const unsigned int component) const
  {
	  (void) component;
    Assert(component == 0, ExcInternalError());
    return 300;
  }


  //----------------------— Cantiliver_beam_3d -------------------------------—
  template <int dim>
  class IncrementalBoundaryForce : public Function<dim>
  {
  public:
	  IncrementalBoundaryForce (const double present_time,
                            const double end_time);

	  virtual
	  void vector_value (const Point<dim> &p,
                     Vector<double> &values) const;

	  virtual
	  void
	  vector_value_list (const std::vector<Point<dim> > &points,
                     std::vector<Vector<double> >   &value_list) const;

  private:
	  const double present_time,
	  	  	  end_time,
			  pressure,
			  height;
  };

  template <int dim>
  IncrementalBoundaryForce<dim>::
  IncrementalBoundaryForce (const double present_time,
                          const double end_time)
						  :
						  Function<dim>(dim),
						  present_time (present_time),
						  end_time (end_time),
						  pressure (6e6),
						  height (2*0.05e-3)
						  {}

  template <int dim>
  void
  IncrementalBoundaryForce<dim>::vector_value (const Point<dim> &/*p*/,
                                             Vector<double> &values) const
  {
	  AssertThrow (dim == 3, ExcNotImplemented());
	  AssertThrow (values.size() == dim,
               ExcDimensionMismatch (values.size(), dim));

//	  const double eps = 1.e-7 * height;

//      AssertThrow(std::abs(p[2]-(height/2)) < eps, ExcInternalError());

	  values = 0;

	  values(1) = -pressure;

	  const double frac = present_time/end_time;

	  values *= frac;

  }

  template <int dim>
  void
  IncrementalBoundaryForce<dim>::
  vector_value_list (const std::vector<Point<dim> > &points,
                   std::vector<Vector<double> >   &value_list) const
				   {
	  const unsigned int n_points = points.size();

	  Assert (value_list.size() == n_points,
			  ExcDimensionMismatch (value_list.size(), n_points));

	  for (unsigned int p=0; p<n_points; ++p)
		  IncrementalBoundaryForce<dim>::vector_value (points[p], value_list[p]);
				   }


  template <int dim>
  class BodyForce :  public ZeroFunction<dim>
  {
  public:
	  BodyForce () : ZeroFunction<dim> (dim) {}
  };


  template <int dim>
  class IncrementalBoundaryValues :  public Function<dim>
  {
  public:
	  IncrementalBoundaryValues (const double present_time,
                             const double end_time);

	  virtual
	  void
	  vector_value (const Point<dim> &p,
                Vector<double>   &values) const;

	  virtual
	  void
	  vector_value_list (const std::vector<Point<dim> > &points,
                     std::vector<Vector<double> >   &value_list) const;

  private:
	  const double present_time,
	  	  	  	  	  	  end_time;
  };


  template <int dim>
  IncrementalBoundaryValues<dim>::
  IncrementalBoundaryValues (const double present_time,
                           const double end_time)
  :
  Function<dim> (dim),
  present_time (present_time),
  end_time (end_time)
  {}


  template <int dim>
  void
  IncrementalBoundaryValues<dim>::
  vector_value (const Point<dim> &/*p*/,
              Vector<double>   &values) const
			  {
	  AssertThrow (values.size() == dim,
               ExcDimensionMismatch (values.size(), dim));
	  AssertThrow (dim == 3, ExcNotImplemented());

	  values = 0.;
			  }


  template <int dim>
  void
  IncrementalBoundaryValues<dim>::
  vector_value_list (const std::vector<Point<dim> > &points,
                   std::vector<Vector<double> >   &value_list) const
				   {
	  const unsigned int n_points = points.size();

	  Assert (value_list.size() == n_points,
          ExcDimensionMismatch (value_list.size(), n_points));

	  for (unsigned int p=0; p<n_points; ++p)
		  IncrementalBoundaryValues<dim>::vector_value (points[p], value_list[p]);
				   }


}
