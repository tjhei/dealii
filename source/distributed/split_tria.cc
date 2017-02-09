// ---------------------------------------------------------------------
// $Id: tria.cc 32807 2014-04-22 15:01:57Z heister $
//
// Copyright (C) 2015 - 2016 by the deal.II authors
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

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/split_tria.h>
#include <deal.II/distributed/tria.h>


DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_MPI
namespace parallel
{
  namespace split
  {

    template <int dim, int spacedim>
    Triangulation<dim,spacedim>::Triangulation (MPI_Comm mpi_communicator,
                                                const typename dealii::Triangulation<dim,spacedim>::MeshSmoothing smooth_grid):
      dealii::parallel::Triangulation<dim,spacedim>(mpi_communicator,smooth_grid,false)
    {}



    template <int dim, int spacedim>
    void Triangulation<dim,spacedim>::partition()
    {
      //dealii::GridTools::partition_triangulation (this->n_subdomains, *this);

      true_subdomain_ids_of_cells.resize(this->n_active_cells());

      // mark artificial cells
      typename parallel::Triangulation<dim,spacedim>::active_cell_iterator
      cell = this->begin_active(),
      endc = this->end();
      {
        // get halo layer of (ghost) cells
        // parallel::shared::Triangulation<dim>::
        std_cxx11::function<bool (const typename Triangulation<dim,spacedim>::active_cell_iterator &)> predicate
          = IteratorFilters::SubdomainEqualTo(this->my_subdomain);

        const std::vector<typename dealii::Triangulation<dim,spacedim>::active_cell_iterator>
        active_halo_layer_vector = GridTools::compute_active_cell_halo_layer<dealii::Triangulation<dim,spacedim> > (*this, predicate);
        std::set<typename Triangulation<dim,spacedim>::active_cell_iterator>
        active_halo_layer(active_halo_layer_vector.begin(), active_halo_layer_vector.end());

        for (unsigned int index=0; cell != endc; cell++, index++)
          {
            if (cell->is_locally_owned() == false &&
                active_halo_layer.find(cell) == active_halo_layer.end())
              cell->set_subdomain_id(numbers::artificial_subdomain_id);
          }
      }


#ifdef DEBUG
      {
        // TODO: check that we have exactly one ghost layer

      }
#endif
    }





//    template <int dim, int spacedim>
//    const std::vector<types::subdomain_id> &
//    Triangulation<dim,spacedim>::get_true_subdomain_ids_of_cells() const
//    {
//      return true_subdomain_ids_of_cells;
//    }



    template <int dim, int spacedim>
    Triangulation<dim,spacedim>::~Triangulation ()
    {}



    template <int dim, int spacedim>
    void
    Triangulation<dim,spacedim>::execute_coarsening_and_refinement ()
    {
      // TODO: okay if called through copy_triangulation, not implemented if not

      dealii::Triangulation<dim,spacedim>::execute_coarsening_and_refinement ();
      //partition();
      //this->update_number_cache ();
    }





    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::
    copy_triangulation (const dealii::Triangulation<dim, spacedim> &other_tria)
    {
      Assert ((dynamic_cast<const dealii::parallel::distributed::Triangulation<dim,spacedim> *>(&other_tria) == NULL),
              ExcMessage("Cannot use this function on parallel::distributed::Triangulation."));


      //mark_cells
      std::vector<bool> user_flags;
      other_tria.save_user_flags(user_flags);

      // TODO: maybe using the flags is not a great idea after all...
      dealii::Triangulation<dim, spacedim> &other_not_const
        =  const_cast<dealii::Triangulation<dim, spacedim> &>(other_tria);
      other_not_const.clear_user_flags ();


      // mark our cells
      typename Triangulation<dim, spacedim>::active_cell_iterator it = other_not_const.begin_active(),
                                                                  endc = other_not_const.end();
      for (; it != endc; ++it)
        {
          if (it->subdomain_id() == this->locally_owned_subdomain())
            it->set_user_flag();
        }

      // mark ghost cells
      std_cxx11::function<bool (const typename Triangulation<dim,spacedim>::active_cell_iterator &)> predicate
        = IteratorFilters::SubdomainEqualTo(this->locally_owned_subdomain());
      std::vector<typename Triangulation<dim,spacedim>::active_cell_iterator>
      active_halo_layer_vector = GridTools::compute_active_cell_halo_layer (other_not_const, predicate);
      for (typename std::vector<Triangulation<dim,spacedim>::active_cell_iterator>::iterator
           it = active_halo_layer_vector.begin(); it != active_halo_layer_vector.end(); ++it)
        (*it)->set_user_flag();

      GridGenerator::create_mesh_from_marked_cells(*this, other_not_const);

      other_not_const.load_user_flags(user_flags);

      partition();
      //this->update_number_cache ();
    }

  }
}

#else

namespace parallel
{
  namespace shared
  {
    template <int dim, int spacedim>
    Triangulation<dim,spacedim>::Triangulation ()
      :
      dealii::parallel::Triangulation<dim,spacedim>(MPI_COMM_SELF)
    {
      Assert (false, ExcNotImplemented());
    }



    template <int dim, int spacedim>
    bool
    Triangulation<dim,spacedim>::with_artificial_cells() const
    {
      Assert (false, ExcNotImplemented());
      return true;
    }



    template <int dim, int spacedim>
    const std::vector<unsigned int> &
    Triangulation<dim,spacedim>::get_true_subdomain_ids_of_cells() const
    {
      Assert (false, ExcNotImplemented());
      return true_subdomain_ids_of_cells;
    }

  }
}


#endif


/*-------------- Explicit Instantiations -------------------------------*/
#include "split_tria.inst"

DEAL_II_NAMESPACE_CLOSE
