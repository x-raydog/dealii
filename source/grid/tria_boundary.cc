// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria_boundary.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <cmath>

DEAL_II_NAMESPACE_OPEN



/* -------------------------- Boundary --------------------- */


template <int dim, int spacedim>
Boundary<dim, spacedim>::~Boundary ()
{}


template <int dim, int spacedim>
void
Boundary<dim, spacedim>::
get_intermediate_points_on_line (const typename Triangulation<dim, spacedim>::line_iterator &,
                                 std::vector<Point<spacedim> > &) const
{
  Assert (false, ExcPureFunctionCalled());
}



template <int dim, int spacedim>
void
Boundary<dim, spacedim>::
get_intermediate_points_on_quad (const typename Triangulation<dim, spacedim>::quad_iterator &,
                                 std::vector<Point<spacedim> > &) const
{
  Assert (false, ExcPureFunctionCalled());
}


template <int dim, int spacedim>
void
Boundary<dim,spacedim>::
get_intermediate_points_on_face (const typename Triangulation<dim,spacedim>::face_iterator &face,
                                 std::vector<Point<spacedim> > &points) const
{
  Assert (dim>1, ExcImpossibleInDim(dim));

  switch (dim)
    {
    case 2:
      get_intermediate_points_on_line (face, points);
      break;
    case 3:
      get_intermediate_points_on_quad (face, points);
      break;
    default:
      Assert (false, ExcNotImplemented());
    }
}


template <>
void
Boundary<1,1>::
get_intermediate_points_on_face (const Triangulation<1,1>::face_iterator &,
                                 std::vector<Point<1> > &) const
{
  Assert (false, ExcImpossibleInDim(1));
}


template <>
void
Boundary<1,2>::
get_intermediate_points_on_face (const Triangulation<1,2>::face_iterator &,
                                 std::vector<Point<2> > &) const
{
  Assert (false, ExcImpossibleInDim(1));
}


template <>
void
Boundary<1,3>::
get_intermediate_points_on_face (const Triangulation<1,3>::face_iterator &,
                                 std::vector<Point<3> > &) const
{
  Assert (false, ExcImpossibleInDim(1));
}


template <int dim, int spacedim>
Point<spacedim>
Boundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::line_iterator &line,
                    const Point<spacedim>                                &trial_point) const
{
  return Manifold<dim,spacedim>::template project_to_manifold
         <typename Triangulation<dim, spacedim>::line_iterator>(line, trial_point);
}



template <int dim, int spacedim>
Point<spacedim>
Boundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::quad_iterator &quad,
                    const Point<spacedim>                                &trial_point) const
{
  return Manifold<dim,spacedim>::template
         project_to_manifold<typename Triangulation<dim, spacedim>::quad_iterator>(quad, trial_point);
}



template <int dim, int spacedim>
Point<spacedim>
Boundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::hex_iterator &hex,
                    const Point<spacedim>                                &trial_point) const
{
  return Manifold<dim,spacedim>::template
         project_to_manifold<typename Triangulation<dim, spacedim>::hex_iterator>(hex, trial_point);
}



template <int dim, int spacedim>
const std::vector<Point<1> > &
Boundary<dim,spacedim>::
get_line_support_points (const unsigned int n_intermediate_points) const
{
  if (points.size() <= n_intermediate_points ||
      points[n_intermediate_points].get() == nullptr)
    {
      Threads::Mutex::ScopedLock lock(mutex);
      if (points.size() <= n_intermediate_points)
        points.resize(n_intermediate_points+1);

      // another thread might have created points in the meantime
      if (points[n_intermediate_points].get() == nullptr)
        {
          std::shared_ptr<QGaussLobatto<1> >
          quadrature (new QGaussLobatto<1>(n_intermediate_points+2));
          points[n_intermediate_points] = quadrature;
        }
    }
  return points[n_intermediate_points]->get_points();
}




/* -------------------------- StraightBoundary --------------------- */


template <int dim, int spacedim>
StraightBoundary<dim, spacedim>::StraightBoundary ()
{}


template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
get_new_point_on_line (const typename Triangulation<dim, spacedim>::line_iterator &line) const
{
  return (line->vertex(0) + line->vertex(1)) / 2;
}


namespace
{
  // compute the new midpoint of a quad --
  // either of a 2d cell on a manifold in 3d
  // or of a face of a 3d triangulation in 3d
  template <int dim>
  Point<3>
  compute_new_point_on_quad (const typename Triangulation<dim, 3>::quad_iterator &quad)
  {
    // generate a new point in the middle of
    // the face based on the points on the
    // edges and the vertices.
    //
    // there is a pathological situation when
    // this face is on a straight boundary, but
    // one of its edges and the face behind it
    // are not; if that face is refined first,
    // the new point in the middle of that edge
    // may not be at the same position as
    // quad->line(.)->center() would have been,
    // but would have been moved to the
    // non-straight boundary. We cater to that
    // situation by using existing edge
    // midpoints if available, or center() if
    // not
    //
    // note that this situation can not happen
    // during mesh refinement, as there the
    // edges are refined first and only then
    // the face. thus, the check whether a line
    // has children does not lead to the
    // situation where the new face midpoints
    // have different positions depending on
    // which of the two cells is refined first.
    //
    // the situation where the edges aren't
    // refined happens when a higher order
    // MappingQ requests the midpoint of a
    // face, though, and it is for these cases
    // that we need to have the check available
    //
    // note that the factor of 1/8 for each
    // of the 8 surrounding points isn't
    // chosen arbitrarily. rather, we may ask
    // where the harmonic map would place the
    // point (0,0) if we map the square
    // [-1,1]^2 onto the domain that is
    // described using the 4 vertices and 4
    // edge point points of this quad. we can
    // then discretize the harmonic map using
    // four cells and Q1 elements on each of
    // the quadrants of the square [-1,1]^2
    // and see where the midpoint would land
    // (this is the procedure we choose, for
    // example, in
    // GridGenerator::laplace_solve) and it
    // turns out that it will land at the
    // mean of the 8 surrounding
    // points. whether a discretization of
    // the harmonic map with only 4 cells is
    // adequate is a different question
    // altogether, of course.
    return (quad->vertex(0) + quad->vertex(1) +
            quad->vertex(2) + quad->vertex(3) +
            (quad->line(0)->has_children() ?
             quad->line(0)->child(0)->vertex(1) :
             quad->line(0)->center()) +
            (quad->line(1)->has_children() ?
             quad->line(1)->child(0)->vertex(1) :
             quad->line(1)->center()) +
            (quad->line(2)->has_children() ?
             quad->line(2)->child(0)->vertex(1) :
             quad->line(2)->center()) +
            (quad->line(3)->has_children() ?
             quad->line(3)->child(0)->vertex(1) :
             quad->line(3)->center())               ) / 8;
  }
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
get_new_point_on_quad (const typename Triangulation<dim, spacedim>::quad_iterator &quad) const
{
  return FlatManifold<dim,spacedim>::get_new_point_on_quad(quad);
}


template <>
Point<3>
StraightBoundary<2,3>::
get_new_point_on_quad (const Triangulation<2,3>::quad_iterator &quad) const
{
  return compute_new_point_on_quad<2> (quad);
}



template <>
Point<3>
StraightBoundary<3>::
get_new_point_on_quad (const Triangulation<3>::quad_iterator &quad) const
{
  return compute_new_point_on_quad<3> (quad);
}



template <int dim, int spacedim>
void
StraightBoundary<dim, spacedim>::
get_intermediate_points_on_line (const typename Triangulation<dim, spacedim>::line_iterator &line,
                                 std::vector<Point<spacedim> > &points) const
{
  const unsigned int n=points.size();
  Assert(n>0, ExcInternalError());

  // Use interior points of QGaussLobatto quadrature formula support points
  // for consistency with MappingQ
  const std::vector<Point<1> > &line_points = this->get_line_support_points(n);

  const Point<spacedim> vertices[2] = { line->vertex(0),
                                        line->vertex(1)
                                      };

  for (unsigned int i=0; i<n; ++i)
    {
      const double x = line_points[1+i][0];
      points[i] = (1-x)*vertices[0] + x*vertices[1];
    }
}




template <int dim, int spacedim>
void
StraightBoundary<dim, spacedim>::
get_intermediate_points_on_quad (const typename Triangulation<dim, spacedim>::quad_iterator &,
                                 std::vector<Point<spacedim> > &) const
{
  Assert(false, ExcImpossibleInDim(dim));
}



template <>
void
StraightBoundary<3>::
get_intermediate_points_on_quad (const Triangulation<3>::quad_iterator &quad,
                                 std::vector<Point<3> > &points) const
{
  const unsigned int spacedim = 3;

  const unsigned int n=points.size(),
                     m=static_cast<unsigned int>(std::sqrt(static_cast<double>(n)));
  // is n a square number
  Assert(m*m==n, ExcInternalError());

  const std::vector<Point<1> > &line_points = this->get_line_support_points(m);

  const Point<spacedim> vertices[4] = { quad->vertex(0),
                                        quad->vertex(1),
                                        quad->vertex(2),
                                        quad->vertex(3)
                                      };

  for (unsigned int i=0; i<m; ++i)
    {
      const double y=line_points[1+i][0];
      for (unsigned int j=0; j<m; ++j)
        {
          const double x=line_points[1+j][0];
          points[i*m+j]=((1-x) * vertices[0] +
                         x     * vertices[1]) * (1-y) +
                        ((1-x) * vertices[2] +
                         x     * vertices[3]) * y;
        }
    }
}



template <>
void
StraightBoundary<2,3>::
get_intermediate_points_on_quad (const Triangulation<2,3>::quad_iterator &quad,
                                 std::vector<Point<3> > &points) const
{
  const unsigned int spacedim = 3;

  const unsigned int n=points.size(),
                     m=static_cast<unsigned int>(std::sqrt(static_cast<double>(n)));
  // is n a square number
  Assert(m*m==n, ExcInternalError());

  const std::vector<Point<1> > &line_points = this->get_line_support_points(m);

  const Point<spacedim> vertices[4] = { quad->vertex(0),
                                        quad->vertex(1),
                                        quad->vertex(2),
                                        quad->vertex(3)
                                      };

  for (unsigned int i=0; i<m; ++i)
    {
      const double y=line_points[1+i][0];
      for (unsigned int j=0; j<m; ++j)
        {
          const double x=line_points[1+j][0];
          points[i*m+j]=((1-x) * vertices[0] +
                         x     * vertices[1]) * (1-y) +
                        ((1-x) * vertices[2] +
                         x     * vertices[3]) * y;
        }
    }
}



template <>
Tensor<1,1>
StraightBoundary<1,1>::
normal_vector (const Triangulation<1,1>::face_iterator &,
               const Point<1> &) const
{
  Assert (false, ExcNotImplemented());
  return Tensor<1,1>();
}


template <>
Tensor<1,2>
StraightBoundary<1,2>::
normal_vector (const Triangulation<1,2>::face_iterator &,
               const Point<2> &) const
{
  Assert (false, ExcNotImplemented());
  return Tensor<1,2>();
}


template <>
Tensor<1,3>
StraightBoundary<1,3>::
normal_vector (const Triangulation<1,3>::face_iterator &,
               const Point<3> &) const
{
  Assert (false, ExcNotImplemented());
  return Tensor<1,3>();
}


namespace internal
{
  namespace
  {
    /**
     * Compute the normalized cross product of a set of dim-1 basis
     * vectors.
     */
    Tensor<1,2>
    normalized_alternating_product (const Tensor<1,2> (&basis_vectors)[1])
    {
      Tensor<1,2> tmp = cross_product_2d (basis_vectors[0]);
      return tmp/tmp.norm();
    }



    Tensor<1,3>
    normalized_alternating_product (const Tensor<1,3> ( &)[1])
    {
      // we get here from StraightBoundary<2,3>::normal_vector, but
      // the implementation below is bogus for this case anyway
      // (see the assert at the beginning of that function).
      Assert (false, ExcNotImplemented());
      return Tensor<1,3>();
    }



    Tensor<1,3>
    normalized_alternating_product (const Tensor<1,3> (&basis_vectors)[2])
    {
      Tensor<1,3> tmp = cross_product_3d (basis_vectors[0], basis_vectors[1]);
      return tmp/tmp.norm();
    }

  }
}


template <int dim, int spacedim>
Tensor<1,spacedim>
StraightBoundary<dim,spacedim>::
normal_vector (const typename Triangulation<dim,spacedim>::face_iterator &face,
               const Point<spacedim> &p) const
{
  // I don't think the implementation below will work when dim!=spacedim;
  // in fact, I believe that we don't even have enough information here,
  // because we would need to know not only about the tangent vectors
  // of the face, but also of the cell, to compute the normal vector.
  // Someone will have to think about this some more.
  Assert (dim == spacedim, ExcNotImplemented());

  // in order to find out what the normal vector is, we first need to
  // find the reference coordinates of the point p on the given face,
  // or at least the reference coordinates of the closest point on the
  // face
  //
  // in other words, we need to find a point xi so that f(xi)=||F(xi)-p||^2->min
  // where F(xi) is the mapping. this algorithm is implemented in
  // MappingQ1<dim,spacedim>::transform_real_to_unit_cell but only for cells,
  // while we need it for faces here. it's also implemented in somewhat
  // more generality there using the machinery of the MappingQ1 class
  // while we really only need it for a specific case here
  //
  // in any case, the iteration we use here is a Gauss-Newton's iteration with
  //   xi^{n+1} = xi^n - H(xi^n)^{-1} J(xi^n)
  // where
  //   J(xi) = (grad F(xi))^T (F(xi)-p)
  // and
  //   H(xi) = [grad F(xi)]^T [grad F(xi)]
  // In all this,
  //   F(xi) = sum_v vertex[v] phi_v(xi)
  // We get the shape functions phi_v from an object of type FE_Q<dim-1>(1)

  // we start with the point xi=1/2, xi=(1/2,1/2), ...
  const unsigned int facedim = dim-1;

  Point<facedim> xi;
  for (unsigned int i=0; i<facedim; ++i)
    xi[i] = 1./2;

  const double eps = 1e-12;
  Tensor<1,spacedim> grad_F[facedim];
  unsigned int iteration = 0;
  while (true)
    {
      Point<spacedim> F;
      for (unsigned int v=0; v<GeometryInfo<facedim>::vertices_per_cell; ++v)
        F += face->vertex(v) * GeometryInfo<facedim>::d_linear_shape_function(xi, v);

      for (unsigned int i=0; i<facedim; ++i)
        {
          grad_F[i] = 0;
          for (unsigned int v=0; v<GeometryInfo<facedim>::vertices_per_cell; ++v)
            grad_F[i] += face->vertex(v) *
                         GeometryInfo<facedim>::d_linear_shape_function_gradient(xi, v)[i];
        }

      Tensor<1,facedim> J;
      for (unsigned int i=0; i<facedim; ++i)
        for (unsigned int j=0; j<spacedim; ++j)
          J[i] += grad_F[i][j] * (F-p)[j];

      Tensor<2,facedim> H;
      for (unsigned int i=0; i<facedim; ++i)
        for (unsigned int j=0; j<facedim; ++j)
          for (unsigned int k=0; k<spacedim; ++k)
            H[i][j] += grad_F[i][k] * grad_F[j][k];

      const Tensor<1,facedim> delta_xi = -invert(H) * J;
      xi += delta_xi;
      ++iteration;

      Assert (iteration<10,
              ExcMessage("The Newton iteration to find the reference point "
                         "did not converge in 10 iterations. Do you have a "
                         "deformed cell? (See the glossary for a definition "
                         "of what a deformed cell is. You may want to output "
                         "the vertices of your cell."));

      if (delta_xi.norm() < eps)
        break;
    }

  // so now we have the reference coordinates xi of the point p.
  // we then have to compute the normal vector, which we can do
  // by taking the (normalize) alternating product of all the tangent
  // vectors given by grad_F
  return internal::normalized_alternating_product(grad_F);
}



template <>
void
StraightBoundary<1>::
get_normals_at_vertices (const Triangulation<1>::face_iterator &,
                         Boundary<1,1>::FaceVertexNormals &) const
{
  Assert (false, ExcImpossibleInDim(1));
}

template <>
void
StraightBoundary<1,2>::
get_normals_at_vertices (const Triangulation<1,2>::face_iterator &,
                         Boundary<1,2>::FaceVertexNormals &) const
{
  Assert (false, ExcNotImplemented());
}


template <>
void
StraightBoundary<1,3>::
get_normals_at_vertices (const Triangulation<1,3>::face_iterator &,
                         Boundary<1,3>::FaceVertexNormals &) const
{
  Assert (false, ExcNotImplemented());
}



template <>
void
StraightBoundary<2>::
get_normals_at_vertices (const Triangulation<2>::face_iterator &face,
                         Boundary<2,2>::FaceVertexNormals &face_vertex_normals) const
{
  const Tensor<1,2> tangent = face->vertex(1) - face->vertex(0);
  for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_face; ++vertex)
    // compute normals from tangent
    face_vertex_normals[vertex] = Point<2>(tangent[1],
                                           -tangent[0]);
}

template <>
void
StraightBoundary<2,3>::
get_normals_at_vertices (const Triangulation<2,3>::face_iterator &face,
                         Boundary<2,3>::FaceVertexNormals &face_vertex_normals) const
{
  const Tensor<1,3> tangent = face->vertex(1) - face->vertex(0);
  for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_face; ++vertex)
    // compute normals from tangent
    face_vertex_normals[vertex] = Point<3>(tangent[1],
                                           -tangent[0],0);
  Assert(false, ExcNotImplemented());
}




template <>
void
StraightBoundary<3>::
get_normals_at_vertices (const Triangulation<3>::face_iterator &face,
                         Boundary<3,3>::FaceVertexNormals &face_vertex_normals) const
{
  const unsigned int vertices_per_face = GeometryInfo<3>::vertices_per_face;

  static const unsigned int neighboring_vertices[4][2]=
  { {1,2},{3,0},{0,3},{2,1}};
  for (unsigned int vertex=0; vertex<vertices_per_face; ++vertex)
    {
      // first define the two tangent vectors at the vertex by using the
      // two lines radiating away from this vertex
      const Tensor<1,3> tangents[2]
        = { face->vertex(neighboring_vertices[vertex][0])
            - face->vertex(vertex),
            face->vertex(neighboring_vertices[vertex][1])
            - face->vertex(vertex)
          };

      // then compute the normal by taking the cross product. since the
      // normal is not required to be normalized, no problem here
      face_vertex_normals[vertex] = cross_product_3d(tangents[0], tangents[1]);
    };
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::line_iterator &line,
                    const Point<spacedim>  &trial_point) const
{
  return Manifold<dim,spacedim>::project_to_manifold(line, trial_point);
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::quad_iterator &quad,
                    const Point<spacedim>  &y) const
{
  return Manifold<dim,spacedim>::project_to_manifold(quad, y);
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::hex_iterator &hex_iterator,
                    const Point<spacedim>                                &trial_point) const
{
  return Manifold<dim,spacedim>::project_to_manifold(hex_iterator, trial_point);
}


template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_manifold(const std::vector<Point<spacedim> > &/*surrounding_points*/,
                    const Point<spacedim> &trial_point) const
{
  return trial_point;
}


// explicit instantiations
#include "tria_boundary.inst"

DEAL_II_NAMESPACE_CLOSE
