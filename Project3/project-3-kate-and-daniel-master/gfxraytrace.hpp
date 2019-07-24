///////////////////////////////////////////////////////////////////////////////
// gfxraytrace.hpp
//
// Basic ray tracer.
//
// This header includes the following classes, which are interrelated:
//
// - scene represents the complete raytracing process, and
//   composes the following classes, each of which is responsible for
//   a smaller part of raytracing:
//
//       - camera defines the viewer's location and orientation
//
//       - viewport defines the dimensions of the view window in both
//         world coordinates and sceen coordinates
//
//       - abstract_projection is an abstract class for a projection,
//         which can generate view rays
//
//           - orthographic_projection is an abstract_projection that
//             performs orthographic projection, with parallel view
//             rays
//
//           - perspective_projection is an abstract_projection that
//             performs perspective-correct projection, where all view
//             rays originate from the camera location
//
//       - abstract_shader is an abstract class for a shader, which
//         computes the color of a pixel based on a ray-object
//         itersection
//
//           - flat shader just passes object color through and does
//             not take lighting into account
//
//           - blinn_phong_shader uses the Blinn-Phong illumination
//             model to combine ambient light, diffuse light, and
//             specular highlights
//
// - view_ray is a viewing ray, with an origin and direction
//
// - abstract_scene_object is an abstract geometric scene object
//
//       - scene_sphere is a 3D sphere
//
//       - scene_triangle is a 3D triangle
//
//       - scene_mesh is a collection of scene_triangle objects
//
// - point_light is a light located at a specific location
//
// - intersection represents an intersection between a view ray and a
//   scene object
//
// - scene represents a scene that may be rendered, including lights,
//   scene objects, and a background color
//
// This file builds upon gfxnumeric.hpp and gfxalgebra.hpp, so you may want to
// familiarize yourself with those headers before diving into this one.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <limits>

#include <boost/property_tree/json_parser.hpp>

#include "gfxalgebra.hpp"
#include "gfximage.hpp"

namespace gfx {

// Forward declarations of new class types, in alphabetical order.

class abstract_projection;
class abstract_scene_object;
class abstract_shader;
class blinn_phong_shader;
class camera;
class flat_shader;
class intersection;
class orthographic_projection;
class perspective_projection;
class point_light;
class raytracer;
class scene;
class scene_read_exception;
class scene_mesh;
class scene_sphere;
class scene_triangle;
class view_ray;
class viewport;

scene read_scene_json(const std::string& path) noexcept(false);

// Class declarations, in the order that classes are introduced in
// the comment above.

// A scene represents all the geometric information necessary to
// render an image. That includes:
//
// - a background color, used to fill a pixel whose view ray does
//   not intersect any scene object;
//
// - a vector of point lights; and
//
// - a vector of scene objects.
//
// Ordinarily you need at least one light, and many scene objects,
// to make an interesting image. However this is not enforced with
// assertions.
class scene {
public:
  using light_storage_type = std::vector<std::unique_ptr<point_light>>;
  using object_storage_type = std::vector<std::unique_ptr<abstract_scene_object>>;

private:
  std::unique_ptr<camera> camera_;
  std::unique_ptr<viewport> viewport_;
  std::unique_ptr<abstract_projection> projection_;
  std::unique_ptr<abstract_shader> shader_;
  hdr_rgb background_;
  light_storage_type lights_;
  object_storage_type objects_;

public:

  scene(scene&&) noexcept = default;

  scene() noexcept
  : background_(BLACK) {
    assert(!complete());
  }

  // Constructor.
  scene(std::unique_ptr<camera> camera,
        std::unique_ptr<viewport> viewport,
        std::unique_ptr<abstract_projection> projection,
        std::unique_ptr<abstract_shader> shader,
        const hdr_rgb& background) noexcept
  : camera_(std::move(camera)),
    viewport_(std::move(viewport)),
    projection_(std::move(projection)),
    shader_(std::move(shader)),
    background_(background) {

    assert(camera_);
    assert(viewport_);
    assert(projection_);
    assert(shader_);

    assert(complete());
  }

  constexpr bool complete() const noexcept {
    return (camera_ && viewport_ && projection_ && shader_);
  }

  // Accessors.
  const camera& camera() const noexcept {
    assert(camera_);
    return *camera_;
  }
  const viewport& viewport() const noexcept {
    assert(viewport_);
    return *viewport_;
  }
  const abstract_projection& projection() const noexcept {
    assert(projection_);
    return *projection_;
  }
  const abstract_shader& shader() const noexcept {
    assert(shader_);
    return *shader_;
  }
  constexpr const hdr_rgb& background() const noexcept {
    return background_;
  }
  constexpr const light_storage_type& lights() const noexcept {
    return lights_;
  }
   const gfx::point_light& light(int i) const noexcept {
    return *lights_[i];
  }
  constexpr const object_storage_type& objects() const noexcept {
    return objects_;
  }

  // Mutators.
  void camera(std::unique_ptr<gfx::camera> camera) noexcept {
    camera_ = std::move(camera);
  }
  void viewport(std::unique_ptr<gfx::viewport> viewport) noexcept {
    viewport_ = std::move(viewport);
  }
  void projection(std::unique_ptr<abstract_projection> projection) noexcept {
    projection_ = std::move(projection);
  }
  void shader(std::unique_ptr<abstract_shader> shader) noexcept {
    shader_ = std::move(shader);
  }
  void background(const hdr_rgb& background) noexcept {
    background_ = background;
  }

  // Adding lights and objects.
  void add_light(std::unique_ptr<point_light> light) noexcept {
    lights_.emplace_back(std::move(light));
  }
  void add_object(std::unique_ptr<abstract_scene_object> object) noexcept {
    objects_.emplace_back(std::move(object));
  }

  // Trace a ray and find the closest intersecting scene object.
  //
  // If no such intersection exists, return an empty optional.
  //
  // If there is an intersection within that t range, return an optional that
  // contains that intersection object.
  std::optional<intersection> intersect(const view_ray& ray) const noexcept;

  // Render the given scene and return the resulting image.
  hdr_image render() const noexcept;

  // Load a scene from a JSON file; throws scene_read_exception on error.
  static scene read_json(const std::string& path) noexcept(false);
};

// An error encountered while trying to read and parse a scene file.
class scene_read_exception {
private:
  std::string path_;
  std::string message_;

public:
  scene_read_exception(const std::string& path,
                       const std::string& message) noexcept
  : path_(path), message_(message) { }

  const std::string& path() const noexcept { return path_; }
  const std::string& message() const noexcept { return message_; }
};

// The location and orientation of the camera. This is defined by a
// 3D point for the eye location; and a basis defined by three 3D
// normalized vectors.
class camera {
private:
  vector3<double> eye_, u_, v_, w_;

public:

  // Constructor that provides the eye location and three basis
  // vectors directly. u, v, and w must each be normalized (magnitude
  // 1.0).
  constexpr camera(const vector3<double>& eye,
                   const vector3<double>& u,
                   const vector3<double>& v,
                   const vector3<double>& w) noexcept
  : eye_(eye), u_(u), v_(v), w_(w) {

    assert(approx_equal(u.magnitude(), 1.0, .01));
    assert(approx_equal(v.magnitude(), 1.0, .01));
    assert(approx_equal(w.magnitude(), 1.0, .01));
  }

  // Constructor that computes the basis in terms of a given
  // view-direction and up vector.
  constexpr camera(const vector3<double>& eye,
                   const vector3<double>& view_direction,
                   const vector3<double>& up) noexcept;

  // Accessors and mutators.
  constexpr const vector3<double>& eye() const noexcept {
    return eye_;
  }
  constexpr const vector3<double>& u() const noexcept {
    return u_;
  }
  constexpr const vector3<double>& v() const noexcept {
    return v_;
  }
  constexpr const vector3<double>& w() const noexcept {
    return w_;
  }
};

// A viewport defines the boundary of the viewing window. It stores
// the width and height of the image, in screen coordinates; and the
// left, right, top, and bottom of the view window in world
// coordinates.
class viewport {
private:
  size_t x_resolution_, y_resolution_;
  double left_, top_, right_, bottom_;

public:

  // Constructor. The following inequalities must hold:
  //
  // x_resolution, y_resolution > 0
  // left < 0 < right
  // bottom < 0 < top
  //
  constexpr viewport(size_t x_resolution,
                     size_t y_resolution,
                     double left,
                     double top,
                     double right,
                     double bottom) noexcept
    : x_resolution_(x_resolution),
      y_resolution_(y_resolution),
      left_(left),
      top_(top),
      right_(right),
      bottom_(bottom) {
    assert(x_resolution > 0);
    assert(y_resolution > 0);
    assert(left < right);
    assert(bottom < top);
    assert(left < 0.0);
    assert(right > 0.0);
    assert(top > 0.0);
    assert(bottom < 0.0);
  }

  // Accessors.
  constexpr size_t x_resolution() const noexcept {
    return x_resolution_;
  }
  constexpr size_t y_resolution() const noexcept {
    return y_resolution_;
  }
  constexpr double left() const noexcept {
    return left_;
  }
  constexpr double right() const noexcept {
    return right_;
  }
  constexpr double top() const noexcept {
    return top_;
  }
  constexpr double bottom() const noexcept {
    return bottom_;
  }

  // Map an (x, y) screen coordinate to a (u, v) coordinate in
  // [0, 1]^2. Return a 2D vector x, with u in x[0] and v in x[1];
  vector2<double> uv(size_t x, size_t y) const noexcept;
};

// Abstract class defining a projection algorithm.
class abstract_projection {
public:

  // Given a camera and (u, v) coordinate within the viewport,
  // create and return the corresponding viewing ray. (u, v) are
  // expected to come from the camera::uv function.
  virtual view_ray compute_view_ray(const camera& c,
			                              double u,
			                              double v) const noexcept = 0;

  virtual ~abstract_projection() noexcept = default;
};

// Orthographic implementation of abstract_projection.
class orthographic_projection : public abstract_projection {
public:

  constexpr orthographic_projection() noexcept = default;

  virtual view_ray compute_view_ray(const camera& c,
			                              double u,
			                              double v) const noexcept;
};

// Perspective implementation of abstract_projection.
class perspective_projection : public abstract_projection {
private:
  double focal_length_;

public:

  // The perspective projection algorithm needs to know the
  // focal_length of the camera, which is the distance between the
  // eye and the view plane, and must be positive.
  constexpr perspective_projection(double focal_length) noexcept
  : focal_length_(focal_length) {
    assert(focal_length > 0.0);
  }

  // Accessor.
  constexpr double focal_length() const noexcept {
    return focal_length_;
  }

  virtual view_ray compute_view_ray(const camera& c,
			                              double u,
			                              double v) const noexcept;
};

// Abstract class defining a shading algorithm.
class abstract_shader {
public:

  // Given a scene, camera, and particular ray-object intersection,
  // compute the color of the pixel corresponding to the view
  // ray. The pixel's color is returned.
  virtual hdr_rgb shade(const scene& scene,
			                  const camera& camera,
		                    const intersection& xsect) const noexcept = 0;

  virtual ~abstract_shader() noexcept = default;
};

// Flat-shader implementation of abstract_shader.
class flat_shader : public abstract_shader {
public:

  virtual hdr_rgb shade(const scene& scene,
		                    const camera& camera,
		                    const intersection& xsect) const noexcept;
};

// Blin-Phong implementation of abstract_shader.
class blinn_phong_shader : public abstract_shader {
private:

  double ambient_coefficient_;
  hdr_rgb ambient_color_;
  double diffuse_coefficient_, specular_coefficient_;

public:

  // The Blinn-Phong model depends on the following parameters:
  //
  // ambient_coefficient is the multiplier for ambient light, which
  // must be non-negative. When zero there will be no ambient light.
  //
  // ambient_color is the color of ambient light; usually white in
  // daylight.
  //
  // diffuse_coefficient is the multiplier for diffuse light (object
  // color), which must be non-negative. When zero there is no
  // diffuse light, so only ambient and specular light would be
  // visible.
  //
  // specular_coefficient is the multiplier for specular light
  // (speckles/gloss/glare), which must be non-negative. When zero
  // there are no specular highlights so all objects appear matte.
  //
  blinn_phong_shader(double ambient_coefficient,
	                   const hdr_rgb& ambient_color,
	                   double diffuse_coefficient,
	                   double specular_coefficient)
  : ambient_coefficient_(ambient_coefficient),
    ambient_color_(ambient_color),
    diffuse_coefficient_(diffuse_coefficient),
    specular_coefficient_(specular_coefficient) {

    assert(ambient_coefficient >= 0.0);
    assert(diffuse_coefficient >= 0.0);
    assert(specular_coefficient >= 0.0);
  }

  // Accessors.
  constexpr double ambient_coefficient () const noexcept {
    return ambient_coefficient_ ;
  }
  constexpr const hdr_rgb& ambient_color() const noexcept {
    return ambient_color_;
  }
  constexpr double diffuse_coefficient () const noexcept {
    return diffuse_coefficient_;
  }
  constexpr double specular_coefficient() const noexcept {
    return specular_coefficient_;
  }

  virtual hdr_rgb shade(const scene& scene,
		                    const camera& camera,
		                    const intersection& xsect) const noexcept;
};

// A view ray represents a ray traveling from the viewer out into
// the scene. It is defined by an origin, and direction, each of
// which is a 3D vector.

class view_ray {
private:
  vector3<double> origin_, direction_;

public:

  // Constructor with an explicit origin and direction. Direction
  // must be normalized (magnitude 1).
  constexpr view_ray(const vector3<double>& origin,
                     const vector3<double>& direction) noexcept
  : origin_(origin),
    direction_(direction) { }

  // Accessors.
  constexpr const vector3<double>& origin() const noexcept {
    return origin_;
  }
  constexpr const vector3<double>& direction() const noexcept {
    return direction_;
  }
};

// Abstract class for some kind of scene object.
class abstract_scene_object {
private:
  hdr_rgb color_;
  double shininess_;

public:

  // Construct an object with the given diffuse color and shininesss
  // value (Phong exponent). shininess must be positive.
  constexpr abstract_scene_object(const hdr_rgb& color,
		                             double shininess) noexcept
  : color_(color),
    shininess_(shininess) {

    assert(shininess > 0.0);
  }

  virtual ~abstract_scene_object() noexcept = default;

  // Accessors.
  constexpr const hdr_rgb& color() const noexcept {
    return color_;
  }
  constexpr double shininess() const noexcept {
    return shininess_;
  }

  // Virtual function to find the intersection between this object
  // and the given viewing ray, if any.
  //
  // If no such intersection exists, return an empty optional.
  //
  // If there is an intersection within that t range, return an optional that
  // contains that intersection object.
  virtual std::optional<intersection> intersect(const view_ray& ray,
                                                double t_min,
                                                double t_upper_bound) const noexcept = 0;
};

// A scene object that is a 3D sphere.
class scene_sphere : public abstract_scene_object {
private:
  vector3<double> center_;
  double radius_;

public:

  // Create a sphere with the given color, shininess, center
  // location, and radius. radius must be positive.
  constexpr scene_sphere(const hdr_rgb& color,
	                       double shininess,
	                       const vector3<double>& center,
	                       double radius) noexcept
  : abstract_scene_object(color, shininess),
    center_(center),
    radius_(radius) {

    assert(radius > 0.0);
  }

  // Accessors.
  constexpr const vector3<double>& center() const noexcept {
    return center_;
  }
  constexpr double radius() const noexcept {
    return radius_;
  }

  virtual std::optional<intersection> intersect(const view_ray& ray,
                                                double t_min,
                                                double t_upper_bound) const noexcept;
};

// A scene object that is a 3D triangle.
class scene_triangle : public abstract_scene_object {
private:
  vector3<double> a_, b_, c_;

public:

  // The three vertices of the triangle are called a, b, c. Each is
  // a 3D location.
  constexpr scene_triangle(const hdr_rgb& color,
	                         double shininess,
	                         const vector3<double>& a,
	                         const vector3<double>& b,
	                         const vector3<double>& c) noexcept
  : abstract_scene_object(color, shininess),
    a_(a),
    b_(b),
    c_(c) { }

  // Accessors.
  constexpr const vector3<double>& a() const noexcept {
    return a_;
  }
  constexpr const vector3<double>& b() const noexcept {
    return b_;
  }
  constexpr const vector3<double>& c() const noexcept {
    return c_;
  }

  virtual std::optional<intersection> intersect(const view_ray& ray,
                                                double t_min,
                                                double t_upper_bound) const noexcept;
};

// A point_light represents a light source that gives off the same
// amount of light in all directions. The sun, or an idealized light
// bulb, can be modeled as a point light.
class point_light {
private:
  vector3<double> location_;
  hdr_rgb color_;
  double intensity_;

public:

  // Construct a point light at the given location, that emits light
  // of the given color, with the given scalar intensity. Intensity must
  // be positive.
  constexpr point_light(const vector3<double>& location,
	                      const hdr_rgb& color,
                        double intensity) noexcept
  : location_(location),
    color_(color),
    intensity_(intensity) {

    assert(intensity > 0.0);
  }

  // Accessors.
  constexpr const vector3<double>& location () const noexcept {
    return location_;
  }
  constexpr const hdr_rgb& color() const noexcept {
    return color_;
  }
  constexpr double intensity() const noexcept {
    return intensity_;
  }
};

// An intersection represents a place where a view ray hits a
// scene object. It is defined by:
//
// - a non-owning pointer to the object that was hit;
//
// - the 3D point where the hit occurs;
//
// - a normal vector, that is perpendicular to the object at the hit
//   location; and
//
// - the t value where the hit happened relative to the view ray
//   direction, i.e.
//       location == ray.origin + (t * ray.direction)
class intersection {
private:
  const abstract_scene_object *object_; // non-owning pointer
  vector3<double> location_, normal_;
  double t_;

public:

  // Construct an intersection.
  // The object pointer must not be nullptr.
  // The normal must be normalized (magnitude 1).
  constexpr intersection(const abstract_scene_object* object,
	                       const vector3<double>& location,
	                       const vector3<double>& normal,
	                       double t) noexcept
  : object_(object),
    location_(location),
    normal_(normal),
    t_(t) {

    assert(object != nullptr);
    assert(approx_equal(normal.magnitude(), 1.0, .01));
  }

  constexpr const abstract_scene_object& object() const noexcept {
    return *object_;
  }
  constexpr const vector3<double>& location() const noexcept {
    return location_;
  }
  constexpr const vector3<double>& normal() const noexcept {
    return normal_;
  }
  constexpr double t() const noexcept {
    return t_;
  }
};

scene scene::read_json(const std::string& path) noexcept(false) {

  auto fail = [&](const std::string& message) {
    throw scene_read_exception(path, message);
  };

  boost::property_tree::ptree tree;
  try {
    boost::property_tree::read_json(path, tree);
  } catch (boost::property_tree::json_parser_error e) {
    fail("error parsing JSON: " + e.message());
  }

  auto has = [](auto& node, auto& key) {
    return bool(node.find(key) != node.not_found());
  };

  auto get_unsigned = [](auto& node, auto& key) {
    return node.template get<unsigned>(key);
  };

  auto get_double = [](auto& node, auto& key) {
    return node.template get<double>(key);
  };

  auto get_vector3 = [&](auto& node, auto& key) {
    std::vector<double> v;
    try {
      for (auto& child : node.get_child(key)) {
        v.push_back(child.second.template get_value<double>());
      }
    } catch (boost::property_tree::ptree_bad_path e) {
      fail(std::string("required key not present: ") + key);
    } catch (boost::property_tree::ptree_bad_data e) {
      fail(std::string("key does not map to a list of floating-point values: ") + key);
    }
    if (v.size() != 3) {
      fail(std::string("key does not contain exactly three numbers: ") + key);
    }
    return vector3<double>(v.begin(), v.end());
  };

  auto get_rgb = [&](auto& node, const std::string& key) {
    auto v = get_vector3(node, key);
    if (!std::all_of(v.begin(), v.end(), is_hdr_intensity_valid)) {
      throw scene_read_exception(path, "color intensity out of range");
    }
    return hdr_rgb(v[0], v[1], v[2]);
  };

  scene s;
  assert(!s.complete());

  try {
    s.camera(std::make_unique<class camera>(get_vector3(tree, "camera_eye"),
                                            get_vector3(tree, "camera_view"),
                                            get_vector3(tree, "camera_up")));

    s.viewport(std::make_unique<class viewport>(get_unsigned(tree, "x_resolution"),
                                                get_unsigned(tree, "y_resolution"),
                                                get_double(tree, "viewport_left"),
                                                get_double(tree, "viewport_top"),
                                                get_double(tree, "viewport_right"),
                                                get_double(tree, "viewport_bottom")));

    s.background(get_rgb(tree, "background"));

    if (has(tree, "orthographic_projection")) {
      s.projection(std::make_unique<orthographic_projection>());
    } else if (has(tree, "perspective_focal_length")) {
      s.projection(std::make_unique<perspective_projection>(get_double(tree, "perspective_focal_length")));
    } else {
      throw scene_read_exception(path, "must contain either orthographic_projection or perspective_focal_length");
    }

    if (has(tree, "flat_shader")) {
      s.shader(std::make_unique<flat_shader>());
    } else if (has(tree, "blinn_phong_shader")) {

      auto& child = tree.get_child("blinn_phong_shader");
      auto ambient_coefficient = get_double(child, "ambient_coefficient");
      auto ambient_color = get_rgb(child, "ambient_color");
      auto diffuse_coefficient = get_double(child, "diffuse_coefficient");
      auto specular_coefficient = get_double(child, "specular_coefficient");

      if (ambient_coefficient < 0.0) {
        fail("ambient coefficient must be non-negative");
      }
      if (diffuse_coefficient < 0.0) {
        fail("diffuse coefficient must be non-negative");
      }
      if (specular_coefficient < 0.0) {
        fail("specular coefficient must be non-negative");
      }

      s.shader(std::make_unique<blinn_phong_shader>(ambient_coefficient,
                                                    ambient_color,
                                                    diffuse_coefficient,
                                                    specular_coefficient));

    } else {
      throw scene_read_exception(path, "must contain flat_shader or blinn_phong_shader");
    }

    if (has(tree, "lights")) {
      for (auto& child : tree.get_child("lights")) {
        auto& value = child.second;
        auto l = std::make_unique<point_light>(get_vector3(value, "location"),
                                               get_rgb(value, "color"),
                                               get_double(value, "intensity"));
        s.add_light(std::move(l));
      }
    }

    if (has(tree, "spheres")) {
      for (auto& child : tree.get_child("spheres")) {
        auto& value = child.second;
        auto color = get_rgb(value, "color");
        auto shininess = get_double(value, "shininess");
        auto center = get_vector3(value, "center");
        auto radius = get_double(value, "radius");
        if (shininess < 0.0) {
          fail("sphere shininess must be positive");
        }
        if (radius < 0.0) {
          fail("sphere radius must be positive");
        }
        s.add_object(std::make_unique<scene_sphere>(color, shininess, center, radius));
      }
    }

    if (has(tree, "triangles")) {
      for (auto& child : tree.get_child("triangles")) {
        auto& value = child.second;
        auto color = get_rgb(value, "color");
        auto shininess = get_double(value, "shininess");
        auto a = get_vector3(value, "a");
        auto b = get_vector3(value, "b");
        auto c = get_vector3(value, "c");
        s.add_object(std::make_unique<scene_triangle>(color, shininess, a, b, c));
      }
    }

  } catch (boost::property_tree::ptree_bad_path e) {
    fail(std::string("required key not present: ") + e.what());
  } catch (boost::property_tree::ptree_bad_data e) {
    fail(std::string("key is not the proper type: ") + e.what());
  }

  return s;
}

///////////////////////////////////////////////////////////////////////////////
// START OF TODO
///////////////////////////////////////////////////////////////////////////////

std::optional<intersection> scene::intersect(const view_ray& ray) const noexcept {

  std::optional<intersection> best_intersect = std::nullopt;
  std::optional<intersection> temp_intersect = std::nullopt;
  double t_min = 0.0;
  double t_max = std::numeric_limits<double>::infinity();
  for(auto& obj : objects_) {
    temp_intersect = obj->intersect(ray,t_min,t_max);
    if (temp_intersect != std::nullopt) {
      if (temp_intersect->t() < t_max)
        best_intersect = temp_intersect;
        t_max = best_intersect->t();
    }
  }
  return best_intersect;
}

hdr_image scene::render() const noexcept {

  assert(camera_);
  assert(viewport_);
  assert(projection_);
  assert(shader_);
  assert(viewport_->x_resolution() > 0);
  assert(viewport_->y_resolution() > 0);

  size_t w = viewport_->x_resolution(),
         h = viewport_->y_resolution();

  hdr_image result(w, h, background_);
  assert(!result.is_empty());

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      vector2<double> uv = viewport_->uv(x,y);
      view_ray ray = projection_->compute_view_ray(*camera_, uv[0], uv[1]);
      std::optional<intersection> inter = intersect(ray);
      if (inter != std::nullopt)
        result.pixel(x,y,shader_->shade(*this, *camera_, *inter));
      else
        result.pixel(x,y,background_);
    }
  }
  return result;
}

constexpr camera::camera(const vector3<double>& eye,
	                       const vector3<double>& view_direction,
	                       const vector3<double>& up) noexcept {
  eye_ = eye;
  w_ = -(view_direction.normalized());
  u_ = (up.cross(w_)).normalized();
  v_ = (w_.cross(u_)).normalized();
}

vector2<double> viewport::uv(size_t x, size_t y) const noexcept {
  double u = left_ + (right_ - left_)*(x+0.5)/x_resolution_;
  double v = bottom_ + (top_ - bottom_)*(y+0.5)/y_resolution_;
  return vector2<double>({u,v});
}

view_ray orthographic_projection::compute_view_ray(const camera& c,
					                                         double u,
					                                         double v) const noexcept {
  return view_ray(c.eye() + c.u()*u + c.v()*v, -c.w());
}

view_ray perspective_projection::compute_view_ray(const camera& c,
					                                        double u,
					                                        double v) const noexcept {
  return view_ray(c.eye(), c.u()*u + c.v()*v - c.w()*focal_length());
}

hdr_rgb flat_shader::shade(const scene& scene,
			                     const camera& camera,
			                     const intersection& xsect) const noexcept {

  return xsect.object().color();
}

hdr_rgb blinn_phong_shader::shade(const scene& scene,
				                          const camera& camera,
				                          const intersection& xsect) const noexcept {
  // Ambient
  hdr_rgb result = hdr_rgb(xsect.object().color().r()*ambient_color_.r()*ambient_coefficient_, xsect.object().color().g()*ambient_color_.g()*ambient_coefficient_, xsect.object().color().b()*ambient_color_.b()*ambient_coefficient_);
  for(int i = 0; i < scene.lights().size(); i++){
    // Lambertian
      const point_light& light = scene.light(i);
      vector3<double> l = (light.location() - xsect.location()).normalized();
      vector3<double> temp = fmax(0, (xsect.normal()*l));
      hdr_rgb tempA = hdr_rgb(xsect.object().color().r()*diffuse_coefficient()*temp[0], xsect.object().color().g()*diffuse_coefficient()*temp[1], xsect.object().color().b()*diffuse_coefficient()*temp[2]);
      // Blinn
      vector3<double> v = (camera.eye() - xsect.location()).normalized();
      auto h = (l + v).normalized();
      temp = fmax(0, (xsect.normal()*h));
      temp[0] = pow(temp[0], xsect.object().shininess());
      temp[1] = pow(temp[1], xsect.object().shininess());
      temp[2] = pow(temp[2], xsect.object().shininess());
      hdr_rgb tempB = hdr_rgb(specular_coefficient()*temp[0], specular_coefficient()*temp[1],specular_coefficient()*temp[2]);
      // Add them together
      result.assign(result.r()+tempA.r()+tempB.r() > 1.0? 1.0 : result.r()+tempA.r()+tempB.r(),
        result.g()+tempA.g()+tempB.g() > 1.0? 1.0 : result.g()+tempA.g()+tempB.g(),
        result.b()+tempA.b()+tempB.b() > 1.0? 1.0 : result.b()+tempA.b()+tempB.b());
    }
   return result;
 }

std::optional<intersection>
  scene_sphere::intersect(
  const view_ray& ray,
  double t_min,
  double t_upper_bound) const noexcept {

  assert(t_min < t_upper_bound);

  // Find discriminant
  vector3<double> origin_to_center = ray.origin() - center_;
  double a = ray.direction()*ray.direction();
  double c = origin_to_center*origin_to_center - radius_*radius_;
  double b = ray.direction()*(origin_to_center);
  double discriminant = (b*b) - a*c;
  if (discriminant < 0)
    return std::nullopt;
  else {
    // Find t, if t is within [t_min,t_upper_bound], return the intersection
    double t = (( -b - sqrt(discriminant)) / a);
    if (t_min <= t && t <= t_upper_bound) {
      vector3<double> point_ = ray.origin() + (ray.direction()*t);
      vector3<double> normal_ = (point_ - center_) / radius_;
      return intersection(this, point_, normal_, t);
    }
    t = (( -b + sqrt(discriminant)) / a);
    if (t_min <= t && t <= t_upper_bound) {
      vector3<double> point_ = ray.origin() + (ray.direction()*t);
      vector3<double> normal_ = (point_ - center_) / radius_;
      return intersection(this, point_, normal_, t);
    }
  }
  return std::nullopt;
}

std::optional<intersection>
  scene_triangle::intersect(
    const view_ray& ray,
    double t_min,
    double t_upper_bound) const noexcept {

  assert(t_min < t_upper_bound);

  // Setup the matrix, based on the book.
  matrix3x3<double> mat;
  mat[0][0] = a_[0] - b_[0];
  mat[0][1] = a_[0] - c_[0];
  mat[0][2] = ray.direction()[0];
  mat[1][0] = a_[1] - b_[1];
  mat[1][1] = a_[1] - c_[1];
  mat[1][2] = ray.direction()[1];
  mat[2][0] = a_[2] - b_[2];
  mat[2][1] = a_[2] - c_[2];
  mat[2][2] = ray.direction()[2];
  // Setup the vector
  vector3<double> vec;
  vec[0] = a_[0] - ray.origin()[0];
  vec[1] = a_[1] - ray.origin()[1];
  vec[2] = a_[2] - ray.origin()[2];
  vector3<double> byt = mat.solve(vec);

  // Ensure t, beta, and gamma are valid
  if (byt[2] < t_min || byt[2] > t_upper_bound)
    return std::nullopt;
  if (byt[1] < 0 || byt[1] > 1)
    return std::nullopt;
  if (byt[0] < 0 || byt[0] > 1 - byt[1])
    return std::nullopt;

  // Calculate intersection and return it
  vector3<double> point = ray.origin() + ray.direction()*byt[2];
  // Math to find the correct normal
  vector3<double> normal;
  vector3<double> normal_candidate_0 = (c_-a_).cross(b_-a_).normalized();
  vector3<double> normal_candidate_1 = (b_-a_).cross(c_-a_).normalized();
  auto semi_normal = -ray.direction();
  double temp_0 = normal_candidate_0*semi_normal;
  double temp_1 = normal_candidate_1*semi_normal;
  if (abs(temp_0) < abs(temp_1))
    normal = normal_candidate_0;
  else
    normal = normal_candidate_1;

  return std::optional<intersection>(intersection(this, point, normal, byt[2]));
}

} // namespace gfx
