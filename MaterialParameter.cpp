/*
 * MaterialParameter.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: zhiboluo
 */



namespace Parameters
{
using namespace dealii;



struct Heat_Source
  {
    double v;// = 50e-3; //scanning speed = 5 cm/s
    double w;// = 0.5e-3/2.0;	// radius of laser spot
    double P;// = 100;//*1000;		// laser power 1 W = 1 Kg*m^2/s^3 = 1000 g*m^2/s^3
    double Emsvity;// = 0.7;	//emissivity coefficient of laser
//    double I0;// = 2*(1-Emsvity)*P/M_PI/std::pow(w, 2);	// maximum laser heat input
    double line_length;// = 10e-3;

    static void declare_parameters (ParameterHandler &prm);
    void parse_parameters (ParameterHandler &prm);
  };

  void Heat_Source::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection("Heat source");
    {
//    	std::cout<<"I m here -1"<<std::endl;
      prm.declare_entry("v", "50e-3",
                        Patterns::Double(),
                        "scanning speed of the laser/electron beam");
//      std::cout<<"I m here -2"<<std::endl;
      prm.declare_entry("w", "0.25e-3",
                        Patterns::Double(),
                        "radius of laser spot");
//      std::cout<<"I m here -3"<<std::endl;
      prm.declare_entry("P", "100",
                        Patterns::Double(),
                        "laser power");
//      std::cout<<"I m here -4"<<std::endl;
      prm.declare_entry("Emsvity", "0.7",
                        Patterns::Double(),
                        "emissivity coefficient of laser");
//      std::cout<<"I m here -5"<<std::endl;
      prm.declare_entry("line_length", "10e-3",
                        Patterns::Double(),
                        "length of a scanning track");
    }
    prm.leave_subsection();
  }


  void Heat_Source::parse_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection("Heat source");
    {
      v = prm.get_double("v");
      w = prm.get_double("w");
      P = prm.get_double("P");
      Emsvity = prm.get_double("Emsvity");
//      I0 = prm.get_double("I0");
      line_length = prm.get_double("line_length");
    }
    prm.leave_subsection();
  }

  struct Nonlinear_Material_property
    {
	  //**********thermal property****************************
      Vector<double> input_temperature_for_conductivity_solid;
      Vector<double> input_conductivity_solid;
      Vector<double> input_temperature_for_conductivity_liquid;
      Vector<double> input_conductivity_liquid;
      Vector<double> input_temperature_for_conductivity_powder;
      Vector<double> input_conductivity_powder;

      Vector<double> input_temperature_for_convectivity_solid;
      Vector<double> input_convectivity_solid;
      Vector<double> input_temperature_for_convectivity_liquid;
      Vector<double> input_convectivity_liquid;
      Vector<double> input_temperature_for_convectivity_powder;
      Vector<double> input_convectivity_powder;

      Vector<double> input_temperature_for_specific_heat_solid;
      Vector<double> input_specific_heat_solid;
      Vector<double> input_temperature_for_specific_heat_liquid;
      Vector<double> input_specific_heat_liquid;
      Vector<double> input_temperature_for_specific_heat_powder;
      Vector<double> input_specific_heat_powder;

      Vector<double> input_temperature_for_emissivity_solid;
      Vector<double> input_emissivity_solid;
      Vector<double> input_temperature_for_emissivity_liquid;
      Vector<double> input_emissivity_liquid;
      Vector<double> input_temperature_for_emissivity_powder;
      Vector<double> input_emissivity_powder;

      double density;
      double melt_point;

      //**********mechanical property****************************
      Vector<double> input_temperature_for_elastic_modulus_solid;
      Vector<double> input_elastic_modulus_solid;
      Vector<double> input_temperature_for_elastic_modulus_liquid;
      Vector<double> input_elastic_modulus_liquid;
      Vector<double> input_temperature_for_elastic_modulus_powder;
      Vector<double> input_elastic_modulus_powder;

      Vector<double> input_temperature_for_poisson_ratio_solid;
      Vector<double> input_poisson_ratio_solid;
      Vector<double> input_temperature_for_poisson_ratio_liquid;
      Vector<double> input_poisson_ratio_liquid;
      Vector<double> input_temperature_for_poisson_ratio_powder;
      Vector<double> input_poisson_ratio_powder;

      Vector<double> input_temperature_for_thermal_expansion_solid;
      Vector<double> input_thermal_expansion_solid;
      Vector<double> input_temperature_for_thermal_expansion_liquid;
      Vector<double> input_thermal_expansion_liquid;
      Vector<double> input_temperature_for_thermal_expansion_powder;
      Vector<double> input_thermal_expansion_powder;

      Vector<double> input_temperature_for_elastic_limit_solid;
      Vector<double> input_elastic_limit_solid;
      Vector<double> input_temperature_for_elastic_limit_liquid;
      Vector<double> input_elastic_limit_liquid;
      Vector<double> input_temperature_for_elastic_limit_powder;
      Vector<double> input_elastic_limit_powder;

      Vector<double> input_temperature_for_hardening_parameter_solid;
      Vector<double> input_hardening_parameter_solid;
      Vector<double> input_temperature_for_hardening_parameter_liquid;
      Vector<double> input_hardening_parameter_liquid;
      Vector<double> input_temperature_for_hardening_parameter_powder;
      Vector<double> input_hardening_parameter_powder;


      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };

  void Nonlinear_Material_property::declare_parameters (ParameterHandler &prm)
  {
	  prm.enter_subsection("Temperature dependent solid material conductivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_conduct_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("K_solid_" + Utilities::int_to_string(i), "20",
					  Patterns::Double(), "input conductivity for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material conductivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_conduct_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("K_liquid_" + Utilities::int_to_string(i), "20",
					  Patterns::Double(), "input conductivity for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material conductivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_conduct_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("K_powder_" + Utilities::int_to_string(i), "20",
					  Patterns::Double(), "input conductivity for powder");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent solid material convectivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_convect_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("H_solid_" + Utilities::int_to_string(i), "10",
					  Patterns::Double(), "input convectivity for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material convectivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_convect_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("H_liquid_" + Utilities::int_to_string(i), "15",
					  Patterns::Double(), "input convectivity for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material convectivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_convect_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("H_powder_" + Utilities::int_to_string(i), "10",
					  Patterns::Double(), "input convectivity for powder");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent solid material specific heat");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_specific_heat_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("Cp_solid_" + Utilities::int_to_string(i), "470",
					  Patterns::Double(), "input specific heat for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material specific heat");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_specific_heat_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("Cp_liquid_" + Utilities::int_to_string(i), "470",
					  Patterns::Double(), "input specific heat for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material specific heat");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_specific_heat_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("Cp_powder_" + Utilities::int_to_string(i), "470",
					  Patterns::Double(), "input specific heat for powder");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent solid material emissivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_emissive_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("E_solid_" + Utilities::int_to_string(i), "0.54",
					  Patterns::Double(), "input emissivity for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material emissivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_emissive_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("E_liquid_" + Utilities::int_to_string(i), "0.54",
					  Patterns::Double(), "input emissivity for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material emissivity");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_emissive_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("E_powder_" + Utilities::int_to_string(i), "0.54",
					  Patterns::Double(), "input emissivity for powder");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Material property");
	  {
		  prm.declare_entry("density", "7900",
				  Patterns::Double(), "density of the material");
		  prm.declare_entry("melt_point", "1923",
				  Patterns::Double(), "melt point of the material");
	  }
	  prm.leave_subsection();

/************************ mechanical properties start****************************/
	  // elastic_modulus
	  prm.enter_subsection("Temperature dependent solid material elastic_modulus");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_elastic_modulus_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("E_solid_elastic_modulus_" + Utilities::int_to_string(i), "200e9",
					  Patterns::Double(), "input elastic_modulus for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material elastic_modulus");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_elastic_modulus_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("E_liquid_elastic_modulus_" + Utilities::int_to_string(i), "200e9",
					  Patterns::Double(), "input elastic_modulus for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material elastic_modulus");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_elastic_modulus_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("E_powder_elastic_modulus_" + Utilities::int_to_string(i), "20",
					  Patterns::Double(), "input elastic_modulus for powder");
		  }
	  }
	  prm.leave_subsection();

	  // poisson_ratio
	  prm.enter_subsection("Temperature dependent solid material poisson_ratio");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_poisson_ratio_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("v_solid_poisson_ratio_" + Utilities::int_to_string(i), "0.3",
					  Patterns::Double(), "input poisson_ratio for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material poisson_ratio");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_poisson_ratio_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("v_liquid_poisson_ratio_" + Utilities::int_to_string(i), "0.3",
					  Patterns::Double(), "input poisson_ratio for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material poisson_ratio");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_poisson_ratio_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("v_powder_poisson_ratio_" + Utilities::int_to_string(i), "0.3",
					  Patterns::Double(), "input poisson_ratio for powder");
		  }
	  }
	  prm.leave_subsection();

	  // thermal_expansion
	  prm.enter_subsection("Temperature dependent solid material thermal_expansion");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_thermal_expansion_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("a_solid_thermal_expansion_" + Utilities::int_to_string(i), "15e-6",
					  Patterns::Double(), "input thermal_expansion for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material thermal_expansion");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_thermal_expansion_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("a_liquid_thermal_expansion_" + Utilities::int_to_string(i), "15e-6",
					  Patterns::Double(), "input thermal_expansion for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material thermal_expansion");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_thermal_expansion_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("a_powder_thermal_expansion_" + Utilities::int_to_string(i), "15e-6",
					  Patterns::Double(), "input thermal_expansion for powder");
		  }
	  }
	  prm.leave_subsection();

	  // yield_strength
	  prm.enter_subsection("Temperature dependent solid material yield_strength");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_yield_strength_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("y_solid_yield_strength_" + Utilities::int_to_string(i), "520e6",
					  Patterns::Double(), "input yield_strength for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material yield_strength");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_yield_strength_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("y_liquid_yield_strength_" + Utilities::int_to_string(i), "5e6",
					  Patterns::Double(), "input yield_strength for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material yield_strength");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_yield_strength_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("y_powder_yield_strength_" + Utilities::int_to_string(i), "1e4",
					  Patterns::Double(), "input yield_strength for powder");
		  }
	  }
	  prm.leave_subsection();

	  // hardening
	  prm.enter_subsection("Temperature dependent solid material hardening");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_solid_hardening_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("h_solid_hardening_" + Utilities::int_to_string(i), "8e-6",
					  Patterns::Double(), "input hardening for solid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material hardening");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_liquid_hardening_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("h_liquid_hardening_" + Utilities::int_to_string(i), "1e-6",
					  Patterns::Double(), "input hardening for liquid");
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material hardening");
	  {
		  prm.declare_entry("num", "50",
				  Patterns::Integer(), "number of input temperatures");
		  unsigned int num = prm.get_integer("num");
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  prm.declare_entry("T_powder_hardening_" + Utilities::int_to_string(i), "300",
					  Patterns::Double(), "input temperature");
			  prm.declare_entry("h_powder_hardening_" + Utilities::int_to_string(i), "1e-6",
					  Patterns::Double(), "input hardening for powder");
		  }
	  }
	  prm.leave_subsection();

/************************ mechanical properties end ****************************/
  }

  void Nonlinear_Material_property::parse_parameters (ParameterHandler &prm)
  {
	  prm.enter_subsection("Temperature dependent solid material conductivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_conductivity_solid.reinit(num);
		  input_conductivity_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_conductivity_solid[i-1] = prm.get_double("T_solid_conduct_" + Utilities::int_to_string(i));
			  input_conductivity_solid[i-1] = prm.get_double("K_solid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material conductivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_conductivity_liquid.reinit(num);
		  input_conductivity_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_conductivity_liquid[i-1] = prm.get_double("T_liquid_conduct_" + Utilities::int_to_string(i));
			  input_conductivity_liquid[i-1] = prm.get_double("K_liquid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material conductivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_conductivity_powder.reinit(num);
		  input_conductivity_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_conductivity_powder[i-1] = prm.get_double("T_powder_conduct_" + Utilities::int_to_string(i));
			  input_conductivity_powder[i-1] = prm.get_double("K_powder_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent solid material convectivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_convectivity_solid.reinit(num);
		  input_convectivity_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_convectivity_solid[i-1] = prm.get_double("T_solid_convect_" + Utilities::int_to_string(i));
			  input_convectivity_solid[i-1] = prm.get_double("H_solid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material convectivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_convectivity_liquid.reinit(num);
		  input_convectivity_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_convectivity_liquid[i-1] = prm.get_double("T_liquid_convect_" + Utilities::int_to_string(i));
			  input_convectivity_liquid[i-1] = prm.get_double("H_liquid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material convectivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_convectivity_powder.reinit(num);
		  input_convectivity_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_convectivity_powder[i-1] = prm.get_double("T_powder_convect_" + Utilities::int_to_string(i));
			  input_convectivity_powder[i-1] = prm.get_double("H_powder_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent solid material specific heat");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_specific_heat_solid.reinit(num);
		  input_specific_heat_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_specific_heat_solid[i-1] = prm.get_double("T_solid_specific_heat_" + Utilities::int_to_string(i));
			  input_specific_heat_solid[i-1] = prm.get_double("Cp_solid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material specific heat");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_specific_heat_liquid.reinit(num);
		  input_specific_heat_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_specific_heat_liquid[i-1] = prm.get_double("T_liquid_specific_heat_" + Utilities::int_to_string(i));
			  input_specific_heat_liquid[i-1] = prm.get_double("Cp_liquid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material specific heat");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_specific_heat_powder.reinit(num);
		  input_specific_heat_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_specific_heat_powder[i-1] = prm.get_double("T_powder_specific_heat_" + Utilities::int_to_string(i));
			  input_specific_heat_powder[i-1] = prm.get_double("Cp_powder_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent solid material emissivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_emissivity_solid.reinit(num);
		  input_emissivity_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_emissivity_solid[i-1] = prm.get_double("T_solid_emissive_" + Utilities::int_to_string(i));
			  input_emissivity_solid[i-1] = prm.get_double("E_solid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material emissivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_emissivity_liquid.reinit(num);
		  input_emissivity_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_emissivity_liquid[i-1] = prm.get_double("T_liquid_emissive_" + Utilities::int_to_string(i));
			  input_emissivity_liquid[i-1] = prm.get_double("E_liquid_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material emissivity");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_emissivity_powder.reinit(num);
		  input_emissivity_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_emissivity_powder[i-1] = prm.get_double("T_powder_emissive_" + Utilities::int_to_string(i));
			  input_emissivity_powder[i-1] = prm.get_double("E_powder_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Material property");
	  {
		  density = prm.get_double("density");
		  melt_point = prm.get_double("melt_point");
	  }
	  prm.leave_subsection();

/************************ mechanical properties start****************************/
	  // elastic_modulus
	  prm.enter_subsection("Temperature dependent solid material elastic_modulus");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_elastic_modulus_solid.reinit(num);
		  input_elastic_modulus_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_elastic_modulus_solid[i-1] = prm.get_double("T_solid_elastic_modulus_" + Utilities::int_to_string(i));
			  input_elastic_modulus_solid[i-1] = prm.get_double("E_solid_elastic_modulus_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material elastic_modulus");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_elastic_modulus_liquid.reinit(num);
		  input_elastic_modulus_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_elastic_modulus_liquid[i-1] = prm.get_double("T_liquid_elastic_modulus_" + Utilities::int_to_string(i));
			  input_elastic_modulus_liquid[i-1] = prm.get_double("E_liquid_elastic_modulus_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material elastic_modulus");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_elastic_modulus_powder.reinit(num);
		  input_elastic_modulus_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_elastic_modulus_powder[i-1] = prm.get_double("T_powder_elastic_modulus_" + Utilities::int_to_string(i));
			  input_elastic_modulus_powder[i-1] = prm.get_double("E_powder_elastic_modulus_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  // poisson_ratio
	  prm.enter_subsection("Temperature dependent solid material poisson_ratio");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_poisson_ratio_solid.reinit(num);
		  input_poisson_ratio_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_poisson_ratio_solid[i-1] = prm.get_double("T_solid_poisson_ratio_" + Utilities::int_to_string(i));
			  input_poisson_ratio_solid[i-1] = prm.get_double("v_solid_poisson_ratio_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material poisson_ratio");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_poisson_ratio_liquid.reinit(num);
		  input_poisson_ratio_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_poisson_ratio_liquid[i-1] = prm.get_double("T_liquid_poisson_ratio_" + Utilities::int_to_string(i));
			  input_poisson_ratio_liquid[i-1] = prm.get_double("v_liquid_poisson_ratio_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material poisson_ratio");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_poisson_ratio_powder.reinit(num);
		  input_poisson_ratio_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_poisson_ratio_powder[i-1] = prm.get_double("T_powder_poisson_ratio_" + Utilities::int_to_string(i));
			  input_poisson_ratio_powder[i-1] = prm.get_double("v_powder_poisson_ratio_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  // thermal_expansion
	  prm.enter_subsection("Temperature dependent solid material thermal_expansion");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_thermal_expansion_solid.reinit(num);
		  input_thermal_expansion_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_thermal_expansion_solid[i-1] = prm.get_double("T_solid_thermal_expansion_" + Utilities::int_to_string(i));
			  input_thermal_expansion_solid[i-1] = prm.get_double("a_solid_thermal_expansion_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material thermal_expansion");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_thermal_expansion_liquid.reinit(num);
		  input_thermal_expansion_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_thermal_expansion_liquid[i-1] = prm.get_double("T_liquid_thermal_expansion_" + Utilities::int_to_string(i));
			  input_thermal_expansion_liquid[i-1] = prm.get_double("a_liquid_thermal_expansion_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material thermal_expansion");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_thermal_expansion_powder.reinit(num);
		  input_thermal_expansion_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_thermal_expansion_powder[i-1] = prm.get_double("T_powder_thermal_expansion_" + Utilities::int_to_string(i));
			  input_thermal_expansion_powder[i-1] = prm.get_double("a_powder_thermal_expansion_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  // yield_strength
	  prm.enter_subsection("Temperature dependent solid material yield_strength");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_elastic_limit_solid.reinit(num);
		  input_elastic_limit_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_elastic_limit_solid[i-1] = prm.get_double("T_solid_yield_strength_" + Utilities::int_to_string(i));
			  input_elastic_limit_solid[i-1] = prm.get_double("y_solid_yield_strength_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material yield_strength");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_elastic_limit_liquid.reinit(num);
		  input_elastic_limit_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_elastic_limit_liquid[i-1] = prm.get_double("T_liquid_yield_strength_" + Utilities::int_to_string(i));
			  input_elastic_limit_liquid[i-1] = prm.get_double("y_liquid_yield_strength_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material yield_strength");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_elastic_limit_powder.reinit(num);
		  input_elastic_limit_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_elastic_limit_powder[i-1] = prm.get_double("T_powder_yield_strength_" + Utilities::int_to_string(i));
			  input_elastic_limit_powder[i-1] = prm.get_double("y_powder_yield_strength_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  // hardening
	  prm.enter_subsection("Temperature dependent solid material hardening");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_hardening_parameter_solid.reinit(num);
		  input_hardening_parameter_solid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_hardening_parameter_solid[i-1] = prm.get_double("T_solid_hardening_" + Utilities::int_to_string(i));
			  input_hardening_parameter_solid[i-1] = prm.get_double("h_solid_hardening_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent liquid material hardening");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_hardening_parameter_liquid.reinit(num);
		  input_hardening_parameter_liquid.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_hardening_parameter_liquid[i-1] = prm.get_double("T_liquid_hardening_" + Utilities::int_to_string(i));
			  input_hardening_parameter_liquid[i-1] = prm.get_double("h_liquid_hardening_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

	  prm.enter_subsection("Temperature dependent powder material hardening");
	  {
		  unsigned int num = prm.get_integer("num");
		  input_temperature_for_hardening_parameter_powder.reinit(num);
		  input_hardening_parameter_powder.reinit(num);
		  for (unsigned int i = 1; i <= num; i++)
		  {
			  input_temperature_for_hardening_parameter_powder[i-1] = prm.get_double("T_powder_hardening_" + Utilities::int_to_string(i));
			  input_hardening_parameter_powder[i-1] = prm.get_double("h_powder_hardening_" + Utilities::int_to_string(i));
		  }
	  }
	  prm.leave_subsection();

/************************ mechanical properties end ****************************/
  }

  template <int dim>
  struct AllParameters
		  : public Heat_Source,
			public Nonlinear_Material_property
      {
        AllParameters (const std::string &input_file);

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
      };

      template <int dim>
      AllParameters<dim>::AllParameters (const std::string &input_file)
//        :
//        initial_conditions (EulerEquations<dim>::n_components)
      {
          ParameterHandler prm;
          declare_parameters(prm);
          prm.parse_input(input_file);
//          declare_parameters(prm);
          parse_parameters(prm);
      }


      template <int dim>
      void
      AllParameters<dim>::declare_parameters (ParameterHandler &prm)
      {
          Parameters::Heat_Source::declare_parameters (prm);
          Parameters::Nonlinear_Material_property::declare_parameters(prm);
      }


      template <int dim>
      void
      AllParameters<dim>::parse_parameters (ParameterHandler &prm)
      {
    	  Parameters::Heat_Source::parse_parameters (prm);
    	  Parameters::Nonlinear_Material_property::parse_parameters(prm);
      }


      class TempDeptMatProp //: public Function<dim>
      {
      public:
      	  TempDeptMatProp(const Vector<double> input_temperatures, const Vector<double> input_values);

      	  virtual double value (const double temperature_val) const;
      	  virtual void value_list (const Vector<double> temperature_vectors,
      			  	  	  	  	  	  	  	  	  Vector<double> &values_list) const;
      private:
      	  Vector<double> input_temperature_values;
      	  Vector<double> input_correspond_values;
      };

      TempDeptMatProp::TempDeptMatProp(const Vector<double> input_temperatures, const Vector<double> input_values)
      {
      	  input_temperature_values = input_temperatures;
      	  input_correspond_values = input_values;
      }

      double TempDeptMatProp::value (const double temperature_val) const
      {
      	  if (temperature_val <= input_temperature_values[0])
      	  {
      		  return input_correspond_values[0];
      	  }
      	  else if (temperature_val >= input_temperature_values[input_temperature_values.size() - 1])
      	  {
      		  return input_correspond_values[input_temperature_values.size() - 1];
      	  }
          for (unsigned int i = 1; i < input_temperature_values.size(); i++)
          {
        	  if (temperature_val >= input_temperature_values[i -1] && temperature_val <= input_temperature_values[i])
        	  {
        		  if (std::fabs(input_temperature_values[i] - input_temperature_values[i-1]) < 1e-6 )
        		  {
        			  std::cout<<"Input temperature is INVALID!"<<std::endl;
        			  AssertThrow (false, ExcNotImplemented());
        			  break;
        		  }
        		  double return_val = (input_correspond_values[i] - input_correspond_values[i-1])/(input_temperature_values[i] - input_temperature_values[i-1])
        				  *(temperature_val - input_temperature_values[i]) + input_correspond_values[i];
        		  return return_val;
        	  }
          }
          return 0;
      }

      void TempDeptMatProp::value_list (const Vector<double> temperature_vectors,
      		  	  	  	  	  	  	  	  	  Vector<double> &values_list) const
      {
        Assert (values_list.size() == temperature_vectors.size(),
                ExcDimensionMismatch (values_list.size(), temperature_vectors.size()));
        const unsigned int n_points = temperature_vectors.size();
        for (unsigned int p=0; p<n_points; ++p)
        	values_list[p] = TempDeptMatProp::value (temperature_vectors[p]);
      }

      template <int dim>
       class MaterialData
       {
         public:
      	  MaterialData (const AllParameters<dim> &input_para);
//********************** thermal properties ********************
           double get_conductivity (const double input_temperature,
                                             const unsigned int material_id) const;
           double get_convectivity (const double input_temperature,
                                             const unsigned int material_id) const;
           double get_specific_heat (const double input_temperature,
                                             const unsigned int material_id) const;
           double get_emissivity (const double input_temperature,
                                             const unsigned int material_id) const;
//****************************************************

//********************** mechanical properties ********************
           double get_elastic_modulus (const double input_temperature,
                                             const unsigned int material_id) const;
           double get_poisson_ratio (const double input_temperature,
                                             const unsigned int material_id) const;
           double get_thermal_expansion (const double input_temperature,
                                             const unsigned int material_id) const;
           double get_elastic_limit (const double input_temperature,
                                             const unsigned int material_id) const; // sigma_y0, initial yield stress
           double get_hardening_parameter (const double input_temperature,
                                             const unsigned int material_id) const;
//****************************************************
         private:
           //********************** thermal properties ********************
           TempDeptMatProp conductivity_func_solid;
           TempDeptMatProp conductivity_func_liquid;
           TempDeptMatProp conductivity_func_powder;

           TempDeptMatProp convectivity_func_solid;
           TempDeptMatProp convectivity_func_liquid;
           TempDeptMatProp convectivity_func_powder;

           TempDeptMatProp specific_heat_func_solid;
           TempDeptMatProp specific_heat_func_liquid;
           TempDeptMatProp specific_heat_func_powder;

           TempDeptMatProp emissivity_func_solid;
           TempDeptMatProp emissivity_func_liquid;
           TempDeptMatProp emissivity_func_powder;
           //****************************************************

           //********************** mechanical properties ********************
           TempDeptMatProp elastic_modulus_func_solid;
           TempDeptMatProp elastic_modulus_func_liquid;
           TempDeptMatProp elastic_modulus_func_powder;

           TempDeptMatProp poisson_ratio_func_solid;
           TempDeptMatProp poisson_ratio_func_liquid;
           TempDeptMatProp poisson_ratio_func_powder;

           TempDeptMatProp thermal_expansion_func_solid;
           TempDeptMatProp thermal_expansion_func_liquid;
           TempDeptMatProp thermal_expansion_func_powder;

           TempDeptMatProp elastic_limit_func_solid;
           TempDeptMatProp elastic_limit_func_liquid;
           TempDeptMatProp elastic_limit_func_powder;

           TempDeptMatProp hardening_parameter_func_solid;
           TempDeptMatProp hardening_parameter_func_liquid;
           TempDeptMatProp hardening_parameter_func_powder;
           //****************************************************
         };

       template <int dim>
       MaterialData<dim>::MaterialData (const AllParameters<dim> &input_para)
         :
       		conductivity_func_solid(input_para.input_temperature_for_conductivity_solid, input_para.input_conductivity_solid),
       		conductivity_func_liquid(input_para.input_temperature_for_conductivity_liquid, input_para.input_conductivity_liquid),
       		conductivity_func_powder(input_para.input_temperature_for_conductivity_powder, input_para.input_conductivity_powder),

       		convectivity_func_solid(input_para.input_temperature_for_convectivity_solid, input_para.input_convectivity_solid),
       		convectivity_func_liquid(input_para.input_temperature_for_convectivity_liquid, input_para.input_convectivity_liquid),
       		convectivity_func_powder(input_para.input_temperature_for_convectivity_powder, input_para.input_convectivity_powder),

       		specific_heat_func_solid(input_para.input_temperature_for_specific_heat_solid, input_para.input_specific_heat_solid),
       		specific_heat_func_liquid(input_para.input_temperature_for_specific_heat_liquid, input_para.input_specific_heat_liquid),
       		specific_heat_func_powder(input_para.input_temperature_for_specific_heat_powder, input_para.input_specific_heat_powder),

       		emissivity_func_solid(input_para.input_temperature_for_emissivity_solid, input_para.input_emissivity_solid),
       		emissivity_func_liquid(input_para.input_temperature_for_emissivity_liquid, input_para.input_emissivity_liquid),
       		emissivity_func_powder(input_para.input_temperature_for_emissivity_powder, input_para.input_emissivity_powder),

			elastic_modulus_func_solid(input_para.input_temperature_for_elastic_modulus_solid, input_para.input_elastic_modulus_solid),
			elastic_modulus_func_liquid(input_para.input_temperature_for_elastic_modulus_liquid, input_para.input_elastic_modulus_liquid),
			elastic_modulus_func_powder(input_para.input_temperature_for_elastic_modulus_powder, input_para.input_elastic_modulus_powder),

			poisson_ratio_func_solid(input_para.input_temperature_for_poisson_ratio_solid, input_para.input_poisson_ratio_solid),
			poisson_ratio_func_liquid(input_para.input_temperature_for_poisson_ratio_liquid, input_para.input_poisson_ratio_liquid),
			poisson_ratio_func_powder(input_para.input_temperature_for_poisson_ratio_powder, input_para.input_poisson_ratio_powder),

			thermal_expansion_func_solid(input_para.input_temperature_for_thermal_expansion_solid, input_para.input_thermal_expansion_solid),
			thermal_expansion_func_liquid(input_para.input_temperature_for_thermal_expansion_liquid, input_para.input_thermal_expansion_liquid),
			thermal_expansion_func_powder(input_para.input_temperature_for_thermal_expansion_powder, input_para.input_thermal_expansion_powder),

			elastic_limit_func_solid(input_para.input_temperature_for_elastic_limit_solid, input_para.input_elastic_limit_solid),
			elastic_limit_func_liquid(input_para.input_temperature_for_elastic_limit_liquid, input_para.input_elastic_limit_liquid),
			elastic_limit_func_powder(input_para.input_temperature_for_elastic_limit_powder, input_para.input_elastic_limit_powder),

			hardening_parameter_func_solid(input_para.input_temperature_for_hardening_parameter_solid, input_para.input_hardening_parameter_solid),
			hardening_parameter_func_liquid(input_para.input_temperature_for_hardening_parameter_liquid, input_para.input_hardening_parameter_liquid),
			hardening_parameter_func_powder(input_para.input_temperature_for_hardening_parameter_powder, input_para.input_hardening_parameter_powder)
         {
         }

       template <int dim>
         double
         MaterialData<dim>::get_conductivity (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = conductivity_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = conductivity_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = conductivity_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_convectivity (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = convectivity_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = convectivity_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = convectivity_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_specific_heat (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = specific_heat_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = specific_heat_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = specific_heat_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_emissivity (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = emissivity_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = emissivity_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = emissivity_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

// ****** mechanical properties ***********

       template <int dim>
         double
         MaterialData<dim>::get_elastic_modulus (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = elastic_modulus_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = elastic_modulus_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = elastic_modulus_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_poisson_ratio (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = poisson_ratio_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = poisson_ratio_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = poisson_ratio_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_thermal_expansion (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = thermal_expansion_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = thermal_expansion_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = thermal_expansion_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_elastic_limit (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = elastic_limit_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = elastic_limit_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = elastic_limit_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

       template <int dim>
         double
         MaterialData<dim>::get_hardening_parameter (const double input_temperature,
         												const unsigned int material_id) const
         {
         	double return_val = 0;
         	switch (material_id)
         	{
         	case 0:
         	{// 0_solid
         		return_val = hardening_parameter_func_solid.value(input_temperature);
         		break;
         	}
         	case 1:
         	{// 1_liquid
         		return_val = hardening_parameter_func_liquid.value(input_temperature);
         		break;
         	}
         	case 2:
         	{// 2_powder (default material type)
         		return_val = hardening_parameter_func_powder.value(input_temperature);
         		break;
         	}
             default:
               Assert (false,
                       ExcMessage ("Presently, only data for 3 material types is implemented"));
         	}
         	return return_val;
         }

}



