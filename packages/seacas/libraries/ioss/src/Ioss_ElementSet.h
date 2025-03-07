// Copyright(C) 1999-2020, 2022 National Technology & Engineering Solutions
// of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
// NTESS, the U.S. Government retains certain rights in this software.
//
// See packages/seacas/LICENSE for details

#pragma once

#include "ioss_export.h"

#include "Ioss_EntityType.h" // for EntityType, etc
#include "Ioss_Property.h"   // for Property
#include <Ioss_EntitySet.h>  // for EntitySet
#include <cstddef>           // for size_t
#include <cstdint>           // for int64_t
#include <string>            // for string
#include <vector>            // for vector
namespace Ioss {
  class DatabaseIO;
} // namespace Ioss
namespace Ioss {
  class Field;
} // namespace Ioss

namespace Ioss {

  /** \brief A collection of elements.
   */
  class IOSS_EXPORT ElementSet : public EntitySet
  {
  public:
    ElementSet(); // Used for template typing only
    ElementSet(const ElementSet &) = default;
    ElementSet(DatabaseIO *io_database, const std::string &my_name, int64_t number_elements);

    std::string type_string() const override { return "ElementSet"; }
    std::string short_type_string() const override { return "elementlist"; }
    std::string contains_string() const override { return "Element"; }
    EntityType  type() const override { return ELEMENTSET; }

    // Handle implicit properties -- These are calcuated from data stored
    // in the grouping entity instead of having an explicit value assigned.
    // An example would be 'element_block_count' for a region.
    Property get_implicit_property(const std::string &my_name) const override;

    void block_membership(std::vector<std::string> &block_membership) override;

  protected:
    int64_t internal_get_field_data(const Field &field, void *data,
                                    size_t data_size) const override;

    int64_t internal_put_field_data(const Field &field, void *data,
                                    size_t data_size) const override;
  };
} // namespace Ioss
