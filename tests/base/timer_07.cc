// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
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

// Verify that the boost::optional implementation of Timer::last_lap_data and
// Timer::accumulated_wall_time_data works correctly.

#include "../tests.h"
#include <deal.II/base/timer.h>

#include <regex>
// burn computer time

double s = 0.;
void burn (unsigned int n)
{
  for (unsigned int i=0 ; i<n ; ++i)
    {
      for (unsigned int j=1 ; j<100000 ; ++j)
        {
          s += 1./j * i;
        }
    }
}



void
test_timer(Timer &t)
{
  burn(50);

  const double old_wall_time = t.wall_time();
  AssertThrow(old_wall_time > 0., ExcInternalError());
  const double old_cpu_time = t.wall_time();
  AssertThrow(old_cpu_time > 0., ExcInternalError());
  AssertThrow(t.get_total_data().max == 0., ExcInternalError());

  burn(50);
  AssertThrow(t.stop() > 0., ExcInternalError());

  AssertThrow(t.wall_time() > old_wall_time, ExcInternalError());
  AssertThrow(t.cpu_time() > old_cpu_time, ExcInternalError());
  AssertThrow(t.last_wall_time() > 0., ExcInternalError());
  AssertThrow(t.last_cpu_time() > 0., ExcInternalError());
  AssertThrow(t.get_data().min > 0., ExcInternalError());
  AssertThrow(t.get_total_data().min > 0., ExcInternalError());

  t.reset();
  AssertThrow(t.wall_time() == 0., ExcInternalError());
  AssertThrow(t.cpu_time()== 0., ExcInternalError());
  AssertThrow(t.get_total_data().max == 0, ExcInternalError());

  deallog << "OK" << std::endl;
}



int main (int argc, char **argv)
{
  initlog();
  Timer timer;
  std::ostringstream string_stream;

  try
    {
      const Utilities::MPI::MinMaxAvg data = timer.get_accumulated_wall_time_data();
    }
  catch (const ExceptionBase &exc)
    {
      deallog << exc.what() << std::endl;
    }
  try
    {
      const Utilities::MPI::MinMaxAvg data = timer.get_last_lap_data();
    }
  catch (const ExceptionBase &exc)
    {
      deallog << exc.what() << std::endl;
    }

  timer.stop();
  string_stream << "last lap data:" << std::endl;
  timer.print_last_lap_data(string_stream);
  string_stream << '\n';
  string_stream << "accumulated wall time data:" << std::endl;
  timer.print_last_lap_data(string_stream);
  // replace time values with xs

  std::string string_buffer = string_stream.str();
  std::size_t position = string_buffer.find_first_of("0123456789.");
  while (position != std::string::npos)
    {
      string_buffer[position] = 'x';
      position = string_buffer.find_first_of("0123456789.");
    }

  // split string into lines to help deallog
  std::vector<std::string> split_buffer = Utilities::split_string_list(string_buffer,
                                                                       "\\n");
  for (const std::string &line : split_buffer)
    deallog << line << std::endl;

  deallog << "OK" << std::endl;
}
