// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2016 by the deal.II authors
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

#ifndef dealii__level_data_out_h
#define dealii__level_data_out_h



#include <deal.II/base/config.h>
#include <deal.II/numerics/data_out.h>


DEAL_II_NAMESPACE_OPEN


/**
 * A specialized version of DataOut<dim> that supports output of cell
 * and dof data on multigrid levels.
 */
template<int dim>
class LevelDataOut : public DataOut<dim>
{
public:
  /**
   * Construct this output class. We will only output cells with the given
   * @p subdomain_id and between level @p min_lvl and @p max_lvl. The default
   * for @p max_lvl will output up to the finest existing level.
   */
  LevelDataOut (const unsigned int subdomain_id,
                unsigned int min_lvl = 0,
                unsigned int max_lvl = numbers::invalid_unsigned_int)
    :
    subdomain_id (subdomain_id),
    min_lvl(min_lvl),
    max_lvl(max_lvl)
  {
    Assert(min_lvl<=max_lvl, ExcMessage("invalid levels in LvlDataOut"));

  }

  /**
   * Returns the lowest level that will be output.
   */
  unsigned int get_min_level() const
  {
    return min_lvl;
  }

  /**
   * Returns the finest level that will be output. This value might be
   * numbers::invalid_unsigned_int denoting that all levels up to the finest
   * one will be output.
   */
  unsigned int get_max_level() const
  {
    return max_lvl;
  }

  /**
   * Create a Vector<double> for each level with as many entries as
   * cells. This can be used to output cell-wise data using
   * add_data_vector(MGLevelObject<VectorType>, std::string)
   *
   * @param[out] result The filled set of vectors
   */
  void make_level_cell_data_vector(MGLevelObject<Vector<double> > &result)
  {
    // Note: we need to use n_levels, not n_global_levels here!
    const unsigned int real_max_lvl = std::min(this->triangulation->n_levels()-1, max_lvl);
    result.resize(min_lvl, real_max_lvl);
    for (unsigned int lvl = min_lvl; lvl <= real_max_lvl; ++lvl)
      result[lvl].reinit(this->triangulation->n_cells(lvl));
  }

  /**
   * Create a ghosted vector on each level to be used for DoFData for
   * the given @p dof_handler.
   */
  template <typename VectorType>
  void make_level_dof_data_vector(MGLevelObject<VectorType> &result,
                                  const DoFHandler<dim> &dof_handler)
  {
    // Note: we need to use n_global_levels so we construct the same number
    // of parallel vectors on each rank.
    const unsigned int real_max_lvl = std::min(this->triangulation->n_global_levels()-1, max_lvl);
    result.resize(min_lvl, real_max_lvl);
    for (unsigned int lvl = min_lvl; lvl <= real_max_lvl; ++lvl)
      {
        result[lvl].reinit(dof_handler.n_dofs(lvl));
      }
  }

  /**
   * Create a ghosted vector on each level to be used for DoFData for
   * the given @p dof_handler.
   */
  template <typename VectorType>
  void make_level_dof_data_vector(MGLevelObject<VectorType> &result,
                                  const DoFHandler<dim> &dof_handler,
                                  const MPI_Comm mpi_comm)
  {
    // Note: we need to use n_global_levels so we construct the same number
    // of parallel vectors on each rank.
    const unsigned int real_max_lvl = std::min(this->triangulation->n_global_levels(), max_lvl);
    result.resize(min_lvl, real_max_lvl);
    for (unsigned int lvl = min_lvl; lvl < real_max_lvl; ++lvl)
      {
        IndexSet relevant;
        DoFTools::extract_locally_relevant_level_dofs (dof_handler,
                                                       lvl,
                                                       relevant);
        IndexSet owned = dof_handler.locally_owned_mg_dofs(lvl);
        result[lvl].reinit(owned, relevant, mpi_comm);
      }
  }

  /**
   * Return the first cell including non-owned cells (required by our base class).
   */
  virtual typename DataOut<dim>::cell_iterator
  first_cell () const
  {
    typename DataOut<dim>::cell_iterator
    cell = this->dofs->begin(min_lvl);

    if (cell == end_iterator())
      return this->dofs->end();

    return cell;
  }

  /**
   * Return the next cell including non-owned cells (required by our base class).
   */
  virtual typename DataOut<dim>::cell_iterator
  next_cell (const typename DataOut<dim>::cell_iterator &old_cell) const
  {
    if (old_cell != end_iterator())
      {
        typename DataOut<dim>::cell_iterator
        cell = old_cell;
        ++cell;
        if (cell == end_iterator())
          return this->dofs->end();

        return cell;
      }
    else
      return this->dofs->end();
  }

  /**
   * Return the first cell we want to output (required by our base class)
   */
  virtual typename DataOut<dim>::cell_iterator
  first_locally_owned_cell () const
  {
    typename DataOut<dim>::cell_iterator cell = first_cell();
    while ((cell != end_iterator()) &&
           (cell->level_subdomain_id() != subdomain_id))
      ++cell;
    if (cell == end_iterator())
      return this->dofs->end();
    return cell;
  }

  /**
   * Return the next cell we want to output (required by our base class)
   */
  virtual typename DataOut<dim>::cell_iterator
  next_locally_owned_cell (const typename DataOut<dim>::cell_iterator &old_cell) const
  {
    typename DataOut<dim>::cell_iterator cell = next_cell(old_cell);
    while ((cell != end_iterator()) &&
           (cell->level_subdomain_id() != subdomain_id))
      ++cell;
    if (cell == end_iterator())
      return this->dofs->end();
    return cell;
  }


  /**
   * Overwrite the way we compute cell_data-indices compared to our base class.
   * In the base class, a cell-data vector is expected to be n_active_cells()
   * long and the value written is given by cell->active_cell_index(). Here,
   * we use an index counting from first_cell() over all levels.
   */
  virtual
  void
  compute_index_maps (std::vector<std::vector<unsigned int> > &cell_to_patch_index_map,
                      std::vector<std::pair<typename DataOut<dim>::cell_iterator, unsigned int> > &all_cells) const
  {
    // This is the same as in the base class:
    cell_to_patch_index_map.resize (this->triangulation->n_levels());
    for (unsigned int l=0; l<this->triangulation->n_levels(); ++l)
      {
        // max_index is the largest cell->index on level l
        unsigned int max_index = 0;
        for (typename DataOut<dim>::cell_iterator cell=first_locally_owned_cell(); cell != this->triangulation->end();
             cell = next_locally_owned_cell(cell))
          if (static_cast<unsigned int>(cell->level()) == l)
            max_index = std::max (max_index,
                                  static_cast<unsigned int>(cell->index()));

        cell_to_patch_index_map[l].resize (max_index+1,
                                           dealii::DataOutBase::Patch<dim,dim>::no_neighbor);
      }

    {
      // Instead of computing active_cell_index() we just count using first_cell()
      // and next_cell():
      typename DataOut<dim>::cell_iterator counting_cell = this->first_cell();
      unsigned int index = 0;
      typename DataOut<dim>::cell_iterator cell = first_locally_owned_cell();
      for (; cell != this->triangulation->end();
           cell = next_locally_owned_cell(cell))
        {
          // move forward until counting_cell points at the cell cell we are looking
          // at to compute the current index
          while (counting_cell!=this->end_iterator()
                 && cell != counting_cell)
            {
              ++counting_cell;
              ++index;
            }

          Assert (static_cast<unsigned int>(cell->level()) <
                  cell_to_patch_index_map.size(),
                  ExcInternalError());
          Assert (static_cast<unsigned int>(cell->index()) <
                  cell_to_patch_index_map[cell->level()].size(),
                  ExcInternalError());
          cell_to_patch_index_map[cell->level()][cell->index()] = all_cells.size();
          all_cells.emplace_back (cell, index);
        }
    }

  }


protected:
  /**
   * Helper function returning the end iterator (computed from max_level).
   */
  typename DataOut<dim>::cell_iterator
  end_iterator() const
  {
    if (max_lvl == numbers::invalid_unsigned_int)
      return this->dofs->end();
    else
      return this->dofs->end(max_lvl);
  }
  /**
   * The subdomain_id we want to output on
   */
  const unsigned int subdomain_id;
  /**
   * The minimum level where cells are output from.
   */
  const unsigned int min_lvl;
  /**
   * The maximum level where cells are output from.
   */
  const unsigned int max_lvl;

};





DEAL_II_NAMESPACE_CLOSE

#endif
