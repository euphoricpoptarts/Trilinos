// Copyright(C) 1999-2020, 2022 National Technology & Engineering Solutions
// of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
// NTESS, the U.S. Government retains certain rights in this software.
//
// See packages/seacas/LICENSE for details

#pragma once

#include "iotm_export.h"

#include <Ioss_CodeTypes.h>
#include <Ioss_DBUsage.h>    // for DatabaseUsage
#include <Ioss_DatabaseIO.h> // for DatabaseIO
#include <Ioss_IOFactory.h>  // for IOFactory
#include <Ioss_Map.h>        // for Map

#include <cstddef> // for size_t
#include <cstdint> // for int64_t
#include <string>  // for string
#include <vector>  // for vector

#include "Ioss_State.h" // for State

namespace Iotm {
  class TextMesh;
} // namespace Iotm
namespace Ioss {
  class CommSet;
  class EdgeBlock;
  class EdgeSet;
  class ElementBlock;
  class ElementSet;
  class FaceBlock;
  class FaceSet;
  class Field;
  class GroupingEntity;
  class NodeBlock;
  class NodeSet;
  class PropertyManager;
  class Region;
  class SideBlock;
  class SideSet;
  class StructuredBlock;
} // namespace Ioss

namespace Ioss {
  class EntityBlock;
} // namespace Ioss

/** \brief A namespace for the generated database format.
 */
namespace Iotm {
  class IOTM_EXPORT IOFactory : public Ioss::IOFactory
  {
  public:
    static const IOFactory *factory();

  private:
    IOFactory();
    Ioss::DatabaseIO *make_IO(const std::string &filename, Ioss::DatabaseUsage db_usage,
                              Ioss_MPI_Comm                communicator,
                              const Ioss::PropertyManager &props) const override;
  };

  class IOTM_EXPORT DatabaseIO : public Ioss::DatabaseIO
  {
  public:
    DatabaseIO(Ioss::Region *region, const std::string &filename, Ioss::DatabaseUsage db_usage,
               Ioss_MPI_Comm communicator, const Ioss::PropertyManager &props);
    DatabaseIO(const DatabaseIO &from)            = delete;
    DatabaseIO &operator=(const DatabaseIO &from) = delete;

    ~DatabaseIO() override;

    const std::string get_format() const override { return "TextMesh"; }

    // Check capabilities of input/output database...  Returns an
    // unsigned int with the supported Ioss::EntityTypes or'ed
    // together. If "return_value & Ioss::EntityType" is set, then the
    // database supports that type (e.g. return_value & Ioss::FACESET)
    unsigned entity_field_support() const override;

    int int_byte_size_db() const override { return int_byte_size_api(); }

    const TextMesh *get_text_mesh() const { return m_textMesh; }

    void set_text_mesh(Iotm::TextMesh *textMesh) { m_textMesh = textMesh; }

  private:
    void read_meta_data__() override;

    bool begin__(Ioss::State state) override;
    bool end__(Ioss::State state) override;

    bool begin_state__(int state, double time) override;

    void get_step_times__() override;
    void get_nodeblocks();
    void get_elemblocks();
    void get_nodesets();
    void get_sidesets();
    void get_commsets();
    void get_assemblies();

    const Ioss::Map &get_node_map() const;
    const Ioss::Map &get_element_map() const;

    int64_t get_field_internal(const Ioss::Region *reg, const Ioss::Field &field, void *data,
                               size_t data_size) const override;
    int64_t get_field_internal(const Ioss::NodeBlock *nb, const Ioss::Field &field, void *data,
                               size_t data_size) const override;
    int64_t get_field_internal(const Ioss::ElementBlock *eb, const Ioss::Field &field, void *data,
                               size_t data_size) const override;
    int64_t get_field_internal(const Ioss::SideBlock *ef_blk, const Ioss::Field &field, void *data,
                               size_t data_size) const override;
    int64_t get_field_internal(const Ioss::NodeSet *ns, const Ioss::Field &field, void *data,
                               size_t data_size) const override;
    int64_t get_field_internal(const Ioss::CommSet *cs, const Ioss::Field &field, void *data,
                               size_t data_size) const override;
    int64_t get_field_internal(const Ioss::Assembly *assem, const Ioss::Field &field, void *data,
                               size_t data_size) const override;

    NOOP_GFI(Ioss::EdgeBlock)
    NOOP_GFI(Ioss::FaceBlock)
    NOOP_GFI(Ioss::StructuredBlock)
    NOOP_GFI(Ioss::EdgeSet)
    NOOP_GFI(Ioss::FaceSet)
    NOOP_GFI(Ioss::ElementSet)
    NOOP_GFI(Ioss::SideSet)
    NOOP_GFI(Ioss::Blob)

    // Input only database -- these will never be called...
    NOOP_PFI(Ioss::Region)
    NOOP_PFI(Ioss::NodeBlock)
    NOOP_PFI(Ioss::EdgeBlock)
    NOOP_PFI(Ioss::FaceBlock)
    NOOP_PFI(Ioss::ElementBlock)
    NOOP_PFI(Ioss::StructuredBlock)
    NOOP_PFI(Ioss::SideBlock)
    NOOP_PFI(Ioss::NodeSet)
    NOOP_PFI(Ioss::EdgeSet)
    NOOP_PFI(Ioss::FaceSet)
    NOOP_PFI(Ioss::ElementSet)
    NOOP_PFI(Ioss::SideSet)
    NOOP_PFI(Ioss::CommSet)
    NOOP_PFI(Ioss::Assembly)
    NOOP_PFI(Ioss::Blob)

    void add_transient_fields(Ioss::GroupingEntity *entity);

    TextMesh *m_textMesh{nullptr};

    double currentTime{0.0};
    int    spatialDimension{3};

    int elementBlockCount{0};
    int nodesetCount{0};
    int sidesetCount{0};
    int assemblyCount{0};

    bool m_useVariableDf{true};
  };
} // namespace Iotm
