#include "httplib.h"
#include "hokm.h"
#include "enhanced_player.h"
#include <iostream>
#include <memory>

using namespace hokm;

int main() {
    httplib::Server svr;

    std::cout << "Starting Hokm C++ Web Server on http://localhost:8080..." << std::endl;

    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("<h1>Welcome to Hokm C++</h1><p>The game engine is running.</p>", "text/html");
    });

    svr.Get("/api/start", [](const httplib::Request& req, httplib::Response& res) {
        // Initialize a new game session
        res.set_content("{\"status\": \"started\"}", "application/json");
    });

    svr.Post("/api/play", [](const httplib::Request& req, httplib::Response& res) {
        // Play a card
        res.set_content("{\"status\": \"played\"}", "application/json");
    });

    svr.listen("0.0.0.0", 8080);
    
    return 0;
}
