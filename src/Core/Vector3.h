#pragma once
#include <iostream>

struct Vector3 {
    double x, y, z;

    Vector3() : x(0), y(0), z(0) {}
    Vector3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    Vector3 operator+(const Vector3& other) const { return { x + other.x, y + other.y, z + other.z }; }
    Vector3 operator-(const Vector3& other) const { return { x - other.x, y - other.y, z - other.z }; }
    Vector3 operator*(double scalar) const { return { x * scalar, y * scalar, z * scalar }; }

    double dot(const Vector3& other) const { return x * other.x + y * other.y + z * other.z; }

    Vector3 cross(const Vector3& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    double magnitude() const { return std::sqrt(x * x + y * y + z * z); }
    Vector3 normalized() const {
        double m = magnitude();
        return (m > 0) ? (*this) * (1.0 / m) : Vector3();
    }
};