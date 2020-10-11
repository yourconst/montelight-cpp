// ==Montelight==
// Tegan Brennan, Stephen Merity, Taiyo Wilson

#define _USE_MATH_DEFINES

#include <cmath>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <thread>
#include <chrono>
#include <direct.h>

#include "rapidjson/document.h"

#include "libs/drand.hpp"

#define EPSILON 0.001f

using namespace std;

// Globals
bool EMITTER_SAMPLING = true;
unsigned MAX_DEPTH = 4;
unsigned SAMPLES = 25;

struct Vector {
  double x, y, z;
  //
  Vector(const Vector &o) : x(o.x), y(o.y), z(o.z) {}
  Vector(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
  inline Vector operator+(const Vector &o) const {
    return Vector(x + o.x, y + o.y, z + o.z);
  }
  inline Vector &operator+=(const Vector &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }
  inline Vector operator-(const Vector &o) const {
    return Vector(x - o.x, y - o.y, z - o.z);
  }
  inline Vector operator*(const Vector &o) const {
    return Vector(x * o.x, y * o.y, z * o.z);
  }
  inline Vector operator/(double o) const {
    return Vector(x / o, y / o, z / o);
  }
  inline Vector operator*(double o) const {
    return Vector(x * o, y * o, z * o);
  }
  inline void selfScalar(const int & v) {
      x *= v;
      y *= v;
      z *= v;
  }
  inline double dot(const Vector &o) const {
    return x * o.x + y * o.y + z * o.z;
  }
  inline Vector &norm(){
    return *this = *this * (1 / sqrt(x * x + y * y + z * z));
  }
  inline Vector cross(Vector &o){
    return Vector(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
  }
  inline double min() {
    return fmin(x, fmin(y, z));
  }
  inline double max() {
    return fmax(x, fmax(y, z));
  }
  inline Vector &abs() {
    x = fabs(x); y = fabs(y); z = fabs(z);
    return *this;
  }
  inline Vector &clamp() {
    // C++11 lambda function: http://en.cppreference.com/w/cpp/language/lambda
    auto clampDouble = [](double x) {
      if (x < 0) return 0.0;
      if (x > 1) return 1.0;
      return x;
    };
    x = clampDouble(x); y = clampDouble(y); z = clampDouble(z);
    return *this;
  }
};

struct Ray {
  Vector origin, direction;
  Ray(const Vector &o_, const Vector &d_) : origin(o_), direction(d_) {}
};

struct Image {
  unsigned int width, height, pixelsCount;
  Vector *pixels, *current;
  unsigned int *samples;
  //std::vector<Vector> *raw_samples;
  //
  Image(unsigned int w, unsigned int h) : width(w), height(h) {
    pixelsCount = width * height;
    pixels = new Vector[pixelsCount];
    samples = new unsigned int[pixelsCount];
    //current = new Vector[pixelsCount];
    //raw_samples = new std::vector<Vector>[width * height];
  }
  Vector getPixel(unsigned int x, unsigned int y) {
    unsigned int index = (height - y - 1) * width + x;
    return current[index];
  }
  void setPixel(unsigned int x, unsigned int y, const Vector &v) {
    unsigned int index = (height - y - 1) * width + x;
    pixels[index] += v;
    samples[index] += 1;
    //current[index] = pixels[index] / samples[index];
    //raw_samples[index].push_back(v);
  }
  void setPixelByIndex(unsigned index, const Vector& v) {
      pixels[index] += v;
      samples[index] += 1;
      //current[index] = pixels[index] / samples[index];
  }
  Vector getSurroundingAverage(int x, int y, int pattern=0) {
    unsigned int index = (height - y - 1) * width + x;
    Vector avg;
    int total;
    for (int dy = -1; dy < 2; ++dy) {
      for (int dx = -1; dx < 2; ++dx) {
        if (pattern == 0 && (dx != 0 && dy != 0)) continue;
        if (pattern == 1 && (dx == 0 || dy == 0)) continue;
        if (dx == 0 && dy == 0) {
          continue;
        }
        if (x + dx < 0 || x + dx > width - 1) continue;
        if (y + dy < 0 || y + dy > height - 1) continue;
        index = (height - (y + dy) - 1) * width + (x + dx);
        //avg += current[index];
        avg += pixels[index] / samples[index];
        total += 1;
      }
    }
    return avg / total;
  }
  inline double toInt(double x) {
    return pow(x, 1 / 2.2f) * 255;
  }
  void saveP3(std::string filePrefix) {
    std::string filename = filePrefix + ".ppm";
    std::ofstream f;
    f.open(filename.c_str(), std::ofstream::out);
    // PPM header: P3 => RGB, width, height, and max RGB value
    f << "P3 " << width << " " << height << " " << 255 << std::endl;
    // For each pixel, write the space separated RGB values
    for (int i=0; i < pixelsCount; ++i) {
      auto p = pixels[i] / samples[i];
      unsigned int r = fmin(255, toInt(p.x)), g = fmin(255, toInt(p.y)), b = fmin(255, toInt(p.z));
      f << r << " " << g << " " << b << std::endl;
    }
  }
  void save(std::string filePrefix) {
      std::string filename = filePrefix + ".ppm";
      std::ofstream f;
      f.open(filename.c_str(), std::ofstream::binary);
      // PPM header: P3 => RGB, width, height, and max RGB value
      f << "P6 " << width << " " << height << " " << 255 << std::endl;

      unsigned len = 3 * pixelsCount;
      unsigned char* buf = new unsigned char[len];

      for (int i = 0; i < pixelsCount; ++i) {
          auto p = pixels[i] / samples[i];
          unsigned int r = fmin(255, toInt(p.x)), g = fmin(255, toInt(p.y)), b = fmin(255, toInt(p.z));
          buf[3 * i] = r; buf[3 * i + 1] = g; buf[3 * i + 2] = b;
      }

      f.write((char*)buf, len);
      f.close();
  }
  ~Image() {
    delete[] pixels;
    delete[] samples;
  }
};

struct Shape {
  Vector color, emit;
  double diffusion;
  //
  Shape(const Vector color_, const Vector emit_, const double reflection_ = .0) :
      color(color_), emit(emit_), diffusion(1.0 - reflection_) {}
  virtual double intersects(const Ray &r) const { return 0; }
  virtual Vector randomPoint() const { return Vector(); }
  virtual Vector getNormal(const Vector &p) const { return Vector(); }
};

struct Sphere : Shape {
  Vector center;
  double radius;
  //
  Sphere(
      const Vector center_,
      double radius_,
      const Vector color_,
      const Vector emit_,
      const double reflection_ = .0
  ) :
    Shape(color_, emit_, reflection_), center(center_), radius(radius_) {}
  double intersects(const Ray &r) const {
    // Find if, and at what distance, the ray intersects with this object
    // Equation follows from solving quadratic equation of (r - c) ^ 2
    // http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
    Vector offset = r.origin - center;
    double a = r.direction.dot(r.direction);
    double b = 2 * offset.dot(r.direction);
    double c = offset.dot(offset) - radius * radius;
    // Find discriminant for use in quadratic equation (b^2 - 4ac)
    double disc = b * b - 4 * a * c;
    // If the discriminant is negative, there are no real roots
    // (ray misses sphere)
    if (disc < 0) {
      return 0;
    }
    // The smallest positive root is the closest intersection point
    disc = sqrt(disc);
    double t = - b - disc;
    if (t > EPSILON) {
      return t / 2;
    }
    t = - b + disc;
    if (t > EPSILON) {
      return t / 2;
    }
    return 0;
  }
  Vector randomPoint() const {
    // TODO: Improved methods of random point generation as this is not 100% even
    // See: https://www.jasondavies.com/maps/random-points/
    //
    // Get random spherical coordinates on light
    double theta = drand() * M_PI;
    double phi = drand() * 2 * M_PI;
    // Convert to Cartesian and scale by radius
    double dxr = radius * sin(theta) * cos(phi);
    double dyr = radius * sin(theta) * sin(phi);
    double dzr = radius * cos(theta);
    return Vector(center.x + dxr, center.y + dyr, center.z + dzr);
  }
  Vector getNormal(const Vector &p) const {
    // Point must have collided with surface of sphere which is at radius
    // Normalize the normal by using radius instead of a sqrt call
    return (p - center) / radius;
  }
};

// Set up our testing scenes
// They're all Cornell box inspired: http://graphics.ucsd.edu/~henrik/images/cbox.html
/////////////////////////
// Scene format: Sphere(position, radius, color, emission)
/////////////////////////
std::vector<Shape*> baseScene = {
    // Left sphere
    new Sphere(Vector(1e5 + 1,40.8,81.6), 1e5f, Vector(.75,.25,.25), Vector()),
    // Right sphere
    new Sphere(Vector(-1e5 + 99,40.8,81.6), 1e5f, Vector(.25,.25,.75), Vector()),
    // Back sphere
    new Sphere(Vector(50,40.8, 1e5), 1e5f, Vector(.75,.75,.75), Vector()),
    // Floor sphere
    new Sphere(Vector(50, 1e5, 81.6), 1e5f, Vector(.75,.75,.75), Vector()),
    // Roof sphere
    new Sphere(Vector(50,-1e5 + 81.6,81.6), 1e5f, Vector(.75,.75,.75), Vector())
};
//

struct Scene {
    std::vector<Shape*> objects, lights;
    //
    Scene(const std::vector<Shape*> objects_, const std::vector<Shape*> lights_) :
        objects(objects_), lights(lights_) {}

    Scene(const std::vector<Shape*> objects_) : objects(objects_) {}
};

struct Tracer {
  Scene scene;
  //
  Tracer(const Scene &scene_) : scene(scene_) {}
  std::pair<Shape *, double> getIntersection(const Ray &r) const {
    Shape *hitObj = NULL;
    double closest = 1e20f;
    for (Shape *obj : scene.objects) {
      double distToHit = obj->intersects(r);
      if (distToHit > 0 && distToHit < closest) {
        hitObj = obj;
        closest = distToHit;
      }
    }
    for (Shape* obj : scene.lights) {
        double distToHit = obj->intersects(r);
        if (distToHit > 0 && distToHit < closest) {
            hitObj = obj;
            closest = distToHit;
        }
    }
    return std::make_pair(hitObj, closest);
  }
  Vector getRadiance(const Ray &r, int depth) {
    static const unsigned max_depth = MAX_DEPTH;
    // Work out what (if anything) was hit
    auto result = getIntersection(r);
    Shape *hitObj = result.first;

    if (!hitObj) return Vector();

    Vector hitPos = r.origin + r.direction * result.second;
    Vector norm = hitObj->getNormal(hitPos);
    // Orient the normal according to how the ray struck the object
    if (norm.dot(r.direction) > 0) {
      norm.selfScalar(-1);
      //norm = norm * -1;
    }
    // Work out the contribution from directly sampling the emitters
    Vector lightSampling;
    if (EMITTER_SAMPLING) {
      for (Shape *light : scene.lights) {
        Vector lightPos = light->randomPoint();
        Vector lightDirection = (lightPos - hitPos).norm();
        Ray rayToLight = Ray(hitPos, lightDirection);
        auto lightHit = getIntersection(rayToLight);
        if (light == lightHit.first) {
          double wi = lightDirection.dot(norm);
          if (wi > 0) {
            double srad = 1.5;
            //double srad = 600;
            double cos_a_max = sqrt(1-srad*srad/(hitPos - lightPos).dot(hitPos - lightPos));
            double omega = 2*M_PI*(1-cos_a_max);
            lightSampling += light->emit * wi * omega * M_1_PI;
          }
        }
      }
    }
    // Work out contribution from reflected light
    // Diffuse reflection condition:
    // Create orthogonal coordinate system defined by (x=u, y=v, z=norm)
    double angle = 2 * M_PI * drand();
    double dist_cen = hitObj->diffusion * sqrt(drand());
    Vector u = (fabs(norm.x) > 0.1 ? Vector(0, 1, 0) : Vector(1, 0, 0)).cross(norm).norm();
    Vector v = norm.cross(u);
    // Direction of reflection
    Vector d = (u * cos(angle) * dist_cen + v * sin(angle) * dist_cen + norm * sqrt(1 - dist_cen * dist_cen)).norm();

    // Russian Roulette sampling based on reflectance of material
    //double U = drand();
    //if (depth > 4 /* && (depth > 20 || drand() > 0.5 U > hitObj->color.max())*/) {
    //    return Vector();
    //}
    // Recurse
    Vector reflected = depth >= max_depth ? Vector() : getRadiance(Ray(hitPos, d), depth + 1);
    //

    //if (!EMITTER_SAMPLING || depth == 0) {
    return hitObj->emit + (hitObj->color * lightSampling) * hitObj->diffusion + hitObj->color * reflected;
    //}
    //return hitObj->color * lightSampling + hitObj->color * reflected;
  }
};

struct ThreadInfo {
    double progress = 0;
    bool isDone = false;
};

ThreadInfo* threadsInfo;

int w = 512;
int h = 512;

Image * img;
Scene scene(baseScene);
Tracer tracer = Tracer(scene);

void threadRenderer(int id, int startPos, int lastPos) {
  auto& threadInfo = threadsInfo[id];
  /////////////////////////
  // Variables to modify the process or the images
  //EMITTER_SAMPLING = true;
  //int w = 256, h = 256;
  //int SNAPSHOT_INTERVAL = 10;
  //unsigned int SAMPLES = 50;
  bool FOCUS_EFFECT = false;
  double FOCAL_LENGTH = 50;
  double APERTURE_FACTOR = 1; // ratio of original/new aperture (>1: smaller view angle, <1: larger view angle)
  // Initialize the image
  //Image img(w, h);
  /////////////////////////
  // Set which scene should be raytraced
  //auto &scene = complexScene;
  //Tracer tracer = Tracer(scene);
  /////////////////////////
  // Set up the camera
  double aperture = 0.5135 / APERTURE_FACTOR;
  Vector cx = Vector((w * aperture) / h, 0, 0);
  Vector dir_norm = Vector(0, -0.042612, -1).norm();
  double L = 140;
  double L_new = APERTURE_FACTOR * L;
  double L_diff = L - L_new;
  Vector cam_shift = dir_norm * (L_diff);
  if (L_diff < 0){
    cam_shift = cam_shift * 1.5;
  }
  L = L_new;
  Ray camera = Ray(Vector(50, 52, 295.6) + cam_shift, dir_norm);
  // Cross product gets the vector perpendicular to cx and the "gaze" direction
  Vector cy = (cx.cross(camera.direction)).norm() * aperture;
  /////////////////////////
  // Take a set number of samples per pixel
  
  // For each pixel, sample a ray in that direction

  const float fW = w;
  const float fH = h;

  for (unsigned ind = startPos; ind < lastPos; ++ind) {
    unsigned x = ind % w;
    unsigned y = h - ind / w;

    //Vector& pixel = img->pixels[ind];

    for (unsigned sample = 0; sample < SAMPLES; ++sample) {
        
        /*Vector target = img->getPixel(x, y);
        double A = (target - img->getSurroundingAverage(x, y, sample % 2)).abs().max() / (100 / 255.0);
        if (sample > 10 && drand() > A) {
            continue;
        }
        ++updated;*/
        
        // Jitter pixel randomly in dx and dy according to the tent filter
        double Ux = 2 * drand();
        double Uy = 2 * drand();
        double dx = Ux < 1 ? sqrt(Ux) - 1 : 1 - sqrt(2 - Ux);
        double dy = Uy < 1 ? sqrt(Uy) - 1 : 1 - sqrt(2 - Uy);

        // Calculate the direction of the camera ray
        Vector d = (cx * (((x+dx) / fW) - 0.5)) + (cy * (((y+dy) / fH) - 0.5)) + camera.direction;
        Ray ray = Ray(camera.origin + d * 140, d.norm());
        // If we're actually using depth of field, we need to modify the camera ray to account for that
        if (FOCUS_EFFECT) {
            // Calculate the focal point
            Vector fp = (camera.origin + d * L) + d.norm() * FOCAL_LENGTH;
            // Get a pixel point and new ray direction to calculate where the rays should intersect
            Vector del_x = (cx * dx * L / fW);
            Vector del_y = (cy * dy * L / fH);
            Vector point = camera.origin + d * L;
            point = point + del_x + del_y;
            d = (fp - point).norm();
            ray = Ray(camera.origin + d * L, d.norm());
        }
        // Retrieve the radiance of the given hit location (i.e. brightness of the pixel)
        auto rads = tracer.getRadiance(ray, 0);
        //auto rads = tracer.getRadiance2(ray);
        //pixel += tracer.getRadiance(ray, 0);
        // Clamp the radiance so it is between 0 and 1
        // If we don't do this, antialiasing doesn't work properly on bright lights
        rads.clamp();
        // Add result of sample to image
        img->setPixelByIndex(ind, rads);
    }

    //pixel.selfScalar(1 / SAMPLES);

    threadInfo.progress = double(ind + 1 - startPos) / (lastPos - startPos);
  }

  threadInfo.isDone = true;
}

class IFile {
public:
    std::ifstream stream;
    char* buffer = NULL;
    string str;
    unsigned length;

    IFile(string filename) {
        stream.open(filename, std::ifstream::in | std::ifstream::binary);

        stream.seekg(0, stream.end);
        int len = stream.tellg();
        stream.seekg(0);

        buffer = new char[len];
        stream.read((char*) buffer, len);

        str = string(buffer);

        length = len;
    }

    void clearForJson() {
        str = str.substr(str.find_first_of("{["), str.find_last_of("}]"));
    }

    ~IFile() {
        if (buffer)
            delete[]buffer;

        if (stream.is_open())
            stream.close();
    }
};

double parseDouble(const rapidjson::Value& v, bool allowNull = false) {
    if (v.IsNumber()) return v.GetDouble();

    if (allowNull) return .0;

    throw "Bad double value in json";
}

Vector parseVector(const rapidjson::Value& v, bool allowNull = false) {
//Vector parseVector(const rapidjson::Value &p, char* prop, Vector* dflt = NULL) {
    Vector result;

    /*if (!p.HasMember(prop) && dflt) return *dflt;
    else throw "Has no member";

    auto v = p[prop];*/

    if (v.IsArray()) {
        auto values = v.GetArray();

        result.x = parseDouble(values[0]);
        result.y = parseDouble(values[1]);
        result.z = parseDouble(values[2]);
    }
    else if (v.IsObject()) {
        result.x = parseDouble(v["x"]);
        result.y = parseDouble(v["y"]);
        result.z = parseDouble(v["z"]);
    }
    else if(!allowNull)
        throw "Bad vector in json";

    return result;
}

void parseObjectsArray(
    std::vector<Shape*> & res,
    const rapidjson::Value & v
) {
    if (!v.IsArray()) return;

    int added = 0, skipped = 0;

    auto objs = v.GetArray();

    for (const auto& obj : objs) {
        auto type = obj["type"].GetString();

        //cout << "Shape " << type;

        if (type != (string)"sphere") {
            //cout << endl;
            ++skipped;
            continue;
        }

        res.push_back(new Sphere(
            parseVector(obj["center"]),
            parseDouble(obj["radius"], true),
            parseVector(obj["color"]),
            parseVector(obj["emit"], true),
            parseDouble(obj["reflection"], true)
        ));

        ++added;

        //cout << " added" << endl;
    }

    cout << "Added: " << added << "; Skipped: " << skipped << endl;
}

void parseScene(string filename) {
    IFile file(filename);

    file.clearForJson();

    rapidjson::Document json;
    json.Parse(file.str.c_str());

    auto errors = json.GetParseError();

    cout << "Lights:" << endl;
    parseObjectsArray(tracer.scene.lights, json["lights"]);

    cout << "Objects:" << endl;
    parseObjectsArray(tracer.scene.objects, json["objects"]);
}

unsigned getCurrentTime() {
    return chrono::system_clock::now().time_since_epoch().count();
}

template<typename T> T getValue(string message, T defaultValue) {
    cout << message << " (default - " << defaultValue << "): ";

    std::string input;
    std::getline(std::cin, input);

    if (!input.empty()) {
        std::istringstream stream(input);
        stream >> defaultValue;
    }

    return defaultValue;
}

void binary(void * p, bool need = true) {
    unsigned long long v = *((unsigned long long *) p);
    unsigned sz = 8 * sizeof(v);

    while (--sz + 1)
        cout << bool((1ULL << sz) & v);

    if(need)
        cout << endl;
}

int main(int argc, const char* argv[]) {
    string sceneFileName = getValue("Scene file", (string) "scenes/simple.json");
    parseScene(sceneFileName);

    //string resultFileName = getValue("Result file (w/o extension)", (string) "renders/render");

    w = getValue("Image width", w);
    h = getValue("Image height", w);
    MAX_DEPTH = getValue("Rendering depth", MAX_DEPTH);
    SAMPLES = getValue("Samples count", SAMPLES);
    const int cpus = getValue("Threads count", thread::hardware_concurrency());


    threadsInfo = new ThreadInfo[cpus];

    img = new Image(w, h);

    const unsigned part = img->pixelsCount / cpus;
    unsigned startIndex = 0;

    const unsigned startTime = getCurrentTime();

    for (unsigned i = 0; i < cpus - 1; ++i, startIndex += part) {
        new thread(threadRenderer, i, startIndex, startIndex + part);
    }

    new thread(threadRenderer, cpus - 1, startIndex, img->pixelsCount);

    while (true) {
        double sumProgress = 0;
        bool isAllDone = true;

        for (unsigned i = 0; i < cpus; ++i) {
            sumProgress += threadsInfo[i].progress;
            isAllDone = isAllDone && threadsInfo[i].isDone;
        }

        std::cout << "Progress: "
            << std::fixed << std::setprecision(2) << sumProgress * 100 / cpus
            << "\r" << std::flush;

        if (isAllDone)
            break;

        this_thread::sleep_for(chrono::milliseconds(200));
    }

    unsigned calcTime = (getCurrentTime() - startTime) / 1e4;

    cout << endl << "Time: " << calcTime << " ms" << endl;

    string renderFolder = string("renders/") + sceneFileName.substr(
        sceneFileName.find_last_of('/') + 1,
        sceneFileName.find_last_of('.') - sceneFileName.find_last_of('/') - 1
    );

    _mkdir(renderFolder.c_str());

    // Save the resulting raytraced image

    char resultFileName[100];

    if (w == h) {
        double k = (double) w / 1024;

        sprintf_s(
            resultFileName, "%s/%.3gk d%d s%d t%d",
            renderFolder.c_str(), k, MAX_DEPTH, SAMPLES, calcTime
        );
    } else
        sprintf_s(
            resultFileName, "%s/%dx%d d%d s%d t%d",
            renderFolder.c_str(), w, h, MAX_DEPTH, SAMPLES, calcTime
        );

    cout << resultFileName << endl;

    img->save(resultFileName);

    return 0;
}
