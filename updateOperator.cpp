/*
 * updateOperator.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: zhiboluo
 */




#ifndef POINT_COMPARISON_OPERATOR_H_
#define POINT_COMPARISON_OPERATOR_H_
#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
//#include <deal.II/base/tensor_base.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <cmath>
DEAL_II_NAMESPACE_OPEN
/**
 * Comparison operator overloading. @relates Point
 */
template <int dim, typename Number>
inline
bool operator < (const Point<dim,Number> &p1, const Point<dim,Number> &p2)
{
   unsigned int j=0;
   bool return_val = false;
   for(unsigned int i=0;i<dim;i++)
   {
	   if (p1[i] != p2[i])
	   {
		    return_val = p1[j]<p2[j]?true:false;
		    break;
	   }
	   else
	   {
		   j++;
	   }
   }
   if(j==dim)
   {
	   return_val = false;
//	   return false;
   }
   return return_val;
}


template <int dim, typename Number>
inline
bool operator > (const Point<dim,Number> &p1, const Point<dim,Number> &p2)
{
	unsigned int j=0;
	for(unsigned int i=0;i<dim;i++)
	{
		if (p1[i] != p2[i])
		{
			return p1[j]>p2[j];
		}
		else
		{
			j++;
		}
	}
	if(j==dim)
	{
		return false;
	}
}

template <int dim, typename Number>
inline
bool operator == (const Point<dim,Number> &p1, const Point<dim,Number> &p2)
{
	unsigned int j=0;
	for(unsigned int i=0;i<dim; i++)
	{
		if (p1[i] == p2[i])
	   {
			j++;
	   }
	}
	if (j == dim)
	{
	   return true;
	}
	else
	{
		return false;
	}
}
DEAL_II_NAMESPACE_CLOSE
#endif /* POINT_COMPARISON_OPERATOR_H_ */

