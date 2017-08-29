// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2017 by the deal.II authors
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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef DEAL_II_HAVE_SYS_RESOURCE_H
#  include <sys/resource.h>
#endif

#ifdef DEAL_II_MSVC
#  include <windows.h>
#endif

#include <boost/io/ios_state.hpp>


DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace Timer
  {
    namespace
    {
      /**
       * Type trait for checking whether or not a type is a std::chrono::duration.
       */
      template <typename T>
      struct is_duration : std::false_type {};

      /**
       * Specialization to get the right truth value.
       */
      template <typename Rep, typename Period>
      struct is_duration<std::chrono::duration<Rep, Period>> : std::true_type {};

      /**
       * Convert a double precision number with units of seconds into a
       * specified duration type T. Only valid when T is a
       * std::chrono::duration type.
       */
      template <typename T>
      T
      from_seconds(const double time)
      {
        static_assert(is_duration<T>::value,
                      "The template type should be a duration type.");
        return T(std::lround(T::period::den*(time/T::period::num)));
      }

      /**
       * Convert a given duration into a double precision number with units of
       * seconds.
       */
      template <typename Rep, typename Period>
      double
      to_seconds(const std::chrono::duration<Rep, Period> duration)
      {
        return Period::num*double(duration.count())/Period::den;
      }

      /**
       * Fill a MinMaxAvg struct with default values.
       */
      void
      clear_timing_data(Utilities::MPI::MinMaxAvg &data)
      {
        data.sum = numbers::signaling_nan<double>();
        data.min = numbers::signaling_nan<double>();
        data.max = numbers::signaling_nan<double>();
        data.avg = numbers::signaling_nan<double>();
        data.min_index = numbers::invalid_unsigned_int;
        data.max_index = numbers::invalid_unsigned_int;
      }
    }
  }
}



CPUClock::time_point CPUClock::now() noexcept
{
  double system_cpu_duration = 0.0;
#ifdef DEAL_II_MSVC
  FILETIME cpuTime, sysTime, createTime, exitTime;
  const auto succeeded = GetProcessTimes
  (GetCurrentProcess(), &createTime, &exitTime, &sysTime, &cpuTime);
  if (succeeded)
    {
      system_cpu_duration = (double)
      (((unsigned long long)cpuTime.dwHighDateTime << 32)
      | cpuTime.dwLowDateTime) / 1e7;
    }
  // keep the zero value if GetProcessTimes didn't work
#elif defined(DEAL_II_HAVE_SYS_RESOURCE_H)
  rusage usage;
  getrusage (RUSAGE_SELF, &usage);
  system_cpu_duration = usage.ru_utime.tv_sec + 1.e-6 * usage.ru_utime.tv_usec;
#else
#  warning "Unsupported platform. Porting not finished."
#endif
  return time_point(internal::Timer::from_seconds<duration>(system_cpu_duration));
}



template <typename clock_type_>
Timer::ClockMeasurements<clock_type_>::ClockMeasurements()
  :
  current_lap_start_time(clock_type::now()),
  accumulated_time(duration_type::zero()),
  last_lap_time(duration_type::zero())
{}



template <typename clock_type_>
void
Timer::ClockMeasurements<clock_type_>::reset()
{
  current_lap_start_time = clock_type::now();
  accumulated_time = duration_type::zero();
  last_lap_time = duration_type::zero();
}



Timer::Timer()
  :
  Timer(MPI_COMM_SELF, /*sync_lap_times=*/false)
{}



Timer::Timer(MPI_Comm mpi_communicator,
             const bool sync_lap_times_)
  :
  running (false),
  mpi_communicator (mpi_communicator),
  sync_lap_times(sync_lap_times_)
{
  reset();
  start();
}



void Timer::start ()
{
  running = true;
#ifdef DEAL_II_WITH_MPI
  if (sync_lap_times)
    {
      const int ierr = MPI_Barrier(mpi_communicator);
      AssertThrowMPI(ierr);
    }
#endif
  wall_times.current_lap_start_time = wall_clock_type::now();
  cpu_times.current_lap_start_time = cpu_clock_type::now();
}



double Timer::stop ()
{
  if (running)
    {
      running = false;

      wall_times.last_lap_time = wall_clock_type::now() - wall_times.current_lap_start_time;
      cpu_times.last_lap_time = cpu_clock_type::now() - cpu_times.current_lap_start_time;

      last_lap_wall_time_data = Utilities::MPI::min_max_avg
                                (internal::Timer::to_seconds(wall_times.last_lap_time),
                                 mpi_communicator);
      if (sync_lap_times)
        {
          wall_times.last_lap_time = internal::Timer::from_seconds<decltype(wall_times)::duration_type>
                                     (last_lap_wall_time_data.max);
          cpu_times.last_lap_time = internal::Timer::from_seconds<decltype(cpu_times)::duration_type>
                                    (Utilities::MPI::min_max_avg
                                     (internal::Timer::to_seconds(cpu_times.last_lap_time),
                                      mpi_communicator).max);
        }
      wall_times.accumulated_time += wall_times.last_lap_time;
      cpu_times.accumulated_time += cpu_times.last_lap_time;
      accumulated_wall_time_data = Utilities::MPI::min_max_avg
                                   (internal::Timer::to_seconds(wall_times.accumulated_time),
                                    mpi_communicator);
    }
  return internal::Timer::to_seconds(cpu_times.accumulated_time);
}



double Timer::cpu_time() const
{
  if (running)
    {
      const double running_time = internal::Timer::to_seconds
                                  (cpu_clock_type::now()
                                   - cpu_times.current_lap_start_time
                                   + cpu_times.accumulated_time);
      return Utilities::MPI::sum (running_time, mpi_communicator);
    }
  else
    {
      return Utilities::MPI::sum (internal::Timer::to_seconds(cpu_times.accumulated_time),
                                  mpi_communicator);
    }
}



double Timer::last_cpu_time() const
{
  return internal::Timer::to_seconds(cpu_times.last_lap_time);
}



double Timer::get_lap_time() const
{
  return internal::Timer::to_seconds(wall_times.last_lap_time);
}



double Timer::operator() () const
{
  return cpu_time();
}



double Timer::wall_time () const
{
  wall_clock_type::duration current_elapsed_wall_time;
  if (running)
    current_elapsed_wall_time = wall_clock_type::now()
                                - wall_times.current_lap_start_time
                                + wall_times.accumulated_time;
  else
    current_elapsed_wall_time = wall_times.accumulated_time;

  return internal::Timer::to_seconds(current_elapsed_wall_time);
}



double Timer::last_wall_time () const
{
  return internal::Timer::to_seconds(wall_times.last_lap_time);
}



void Timer::reset ()
{
  wall_times.reset();
  cpu_times.reset();
  running = false;
  internal::Timer::clear_timing_data(last_lap_wall_time_data);
  internal::Timer::clear_timing_data(accumulated_wall_time_data);
}



/* ---------------------------- TimerOutput -------------------------- */

TimerOutput::TimerOutput (std::ostream &stream,
                          const OutputFrequency output_frequency,
                          const OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream, true),
  output_is_enabled (true),
  mpi_communicator (MPI_COMM_SELF)
{}



TimerOutput::TimerOutput (ConditionalOStream &stream,
                          const OutputFrequency output_frequency,
                          const OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream),
  output_is_enabled (true),
  mpi_communicator (MPI_COMM_SELF)
{}



TimerOutput::TimerOutput (MPI_Comm      mpi_communicator,
                          std::ostream &stream,
                          const OutputFrequency output_frequency,
                          const OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream, true),
  output_is_enabled (true),
  mpi_communicator (mpi_communicator)
{}



TimerOutput::TimerOutput (MPI_Comm      mpi_communicator,
                          ConditionalOStream &stream,
                          const OutputFrequency output_frequency,
                          const OutputType output_type)
  :
  output_frequency (output_frequency),
  output_type (output_type),
  out_stream (stream),
  output_is_enabled (true),
  mpi_communicator (mpi_communicator)
{}



TimerOutput::~TimerOutput()
{
  try
    {
      while (active_sections.size() > 0)
        leave_subsection();
    }
  catch (...)
    {}

  if ( (output_frequency == summary || output_frequency == every_call_and_summary)
       && output_is_enabled == true)
    print_summary();
}



void
TimerOutput::enter_subsection (const std::string &section_name)
{
  Threads::Mutex::ScopedLock lock (mutex);

  Assert (section_name.empty() == false,
          ExcMessage ("Section string is empty."));

  Assert (std::find (active_sections.begin(), active_sections.end(),
                     section_name) == active_sections.end(),
          ExcMessage (std::string("Cannot enter the already active section <")
                      + section_name + ">."));

  if (sections.find (section_name) == sections.end())
    {
      if (mpi_communicator != MPI_COMM_SELF)
        {
          // create a new timer for this section. the second argument
          // will ensure that we have an MPI barrier before starting
          // and stopping a timer, and this ensures that we get the
          // maximum run time for this section over all processors.
          // The mpi_communicator from TimerOutput is passed to the
          // Timer here, so this Timer will collect timing information
          // among all processes inside mpi_communicator.
          sections[section_name].timer = Timer(mpi_communicator, true);
        }


      sections[section_name].total_cpu_time = 0;
      sections[section_name].total_wall_time = 0;
      sections[section_name].n_calls = 0;
    }

  sections[section_name].timer.reset();
  sections[section_name].timer.start();
  sections[section_name].n_calls++;

  active_sections.push_back (section_name);
}



void
TimerOutput::leave_subsection (const std::string &section_name)
{
  Assert (!active_sections.empty(),
          ExcMessage("Cannot exit any section because none has been entered!"));

  Threads::Mutex::ScopedLock lock (mutex);

  if (section_name != "")
    {
      Assert (sections.find (section_name) != sections.end(),
              ExcMessage ("Cannot delete a section that was never created."));
      Assert (std::find (active_sections.begin(), active_sections.end(),
                         section_name) != active_sections.end(),
              ExcMessage ("Cannot delete a section that has not been entered."));
    }

  // if no string is given, exit the last
  // active section.
  const std::string actual_section_name = (section_name == "" ?
                                           active_sections.back () :
                                           section_name);

  sections[actual_section_name].timer.stop();
  sections[actual_section_name].total_wall_time
  += sections[actual_section_name].timer.last_wall_time();

  // Get cpu time. On MPI systems, if constructed with an mpi_communicator
  // like MPI_COMM_WORLD, then the Timer will sum up the CPU time between
  // processors among the provided mpi_communicator. Therefore, no
  // communication is needed here.
  const double cpu_time = sections[actual_section_name].timer.last_cpu_time();
  sections[actual_section_name].total_cpu_time += cpu_time;

  // in case we have to print out something, do that here...
  if ((output_frequency == every_call || output_frequency == every_call_and_summary)
      && output_is_enabled == true)
    {
      std::string output_time;
      std::ostringstream cpu;
      cpu << cpu_time << "s";
      std::ostringstream wall;
      wall << sections[actual_section_name].timer.last_wall_time() << "s";
      if (output_type == cpu_times)
        output_time = ", CPU time: " + cpu.str();
      else if (output_type == wall_times)
        output_time = ", wall time: " + wall.str() + ".";
      else
        output_time = ", CPU/wall time: " + cpu.str() + " / " + wall.str() + ".";

      out_stream << actual_section_name << output_time
                 << std::endl;
    }

  // delete the index from the list of
  // active ones
  active_sections.erase (std::find (active_sections.begin(), active_sections.end(),
                                    actual_section_name));
}



std::map<std::string, double>
TimerOutput::get_summary_data (const OutputData kind) const
{
  std::map<std::string, double> output;
  for (const auto &section : sections)
    {
      switch (kind)
        {
        case TimerOutput::OutputData::total_cpu_time:
          output[section.first] = section.second.total_cpu_time;
          break;
        case TimerOutput::OutputData::total_wall_time:
          output[section.first] = section.second.total_wall_time;
          break;
        case TimerOutput::OutputData::n_calls:
          output[section.first] = section.second.n_calls;
          break;
        default:
          Assert(false, ExcNotImplemented());
        }
    }
  return output;
}



void
TimerOutput::print_summary () const
{
  // we are going to change the precision and width of output below. store the
  // old values so we can restore it later on
  boost::io::ios_flags_saver restore_flags(out_stream.get_stream());

  auto print_sections = [&](const OutputType output_type)
    {
      Assert(output_type == cpu_times || output_type == wall_times,
             ExcNotImplemented());
      // generate a nice table
      out_stream << "\n\n"
                 << "+---------------------------------------------+------------"
                 << "+------------+\n";
      switch (output_type)
        {
        case cpu_times:
          out_stream << "| Total CPU time elapsed since start          |";
          break;
        case wall_times:
          out_stream << "| Total wallclock time elapsed since start    |";
          break;
        default:
          Assert(false, ExcNotImplemented());
        }

      double total_time = 0.0;
      double cpu_check_time = 0.0;
      switch (output_type)
        {
        case cpu_times:
          {
            // Check, for the CPU time case, that the sum of all times is
            // less or equal than the total time. otherwise, we might have
            // generated a lot of overhead in this function.
            for (const std::pair<std::string, Section> &section : sections)
              cpu_check_time += section.second.total_cpu_time;

            total_time = std::max(cpu_check_time, timer_all.cpu_time());
            break;
          }
        case wall_times:
          total_time = timer_all.wall_time();
          break;
        default:
          Assert(false, ExcNotImplemented());
        }
      out_stream << std::setw(10) << std::setprecision(3) << std::right
                 << total_time << "s |            |\n"
                 << "|                                             |            "
                 << "|            |\n"
                 << "| Section                         | no. calls |"
                 << std::setw(10);
      switch (output_type)
        {
        case cpu_times:
          out_stream << "  CPU time  | % of total |\n";
          break;
        case wall_times:
          out_stream << "  wall time | % of total |\n";
          break;
        default:
          Assert(false, ExcNotImplemented());
        }
      out_stream << "+---------------------------------+-----------+------------"
                 << "+------------+";

      for (const std::pair<std::string, Section> &section : sections)
        {
          std::string name_out = section.first;
          const Section &current_section = section.second;

          // resize the array so that it is always of the same size
          const std::size_t pos_non_space = name_out.find_first_not_of (' ');
          name_out.erase(0, pos_non_space);
          name_out.resize (32, ' ');
          out_stream << '\n';
          out_stream << "| " << name_out << "| ";
          out_stream << std::setw(9);
          out_stream << current_section.n_calls << " |";
          out_stream << std::setw(10);
          out_stream << std::setprecision(3);
          switch (output_type)
            {
            case cpu_times:
              out_stream << current_section.total_cpu_time << "s |";
              break;
            case wall_times:
              out_stream << current_section.total_wall_time << "s |";
              break;
            default:
              Assert(false, ExcNotImplemented());
            }
          out_stream << std::setw(10);

          if (total_time != 0)
            {
              // If the current section time is less than 0.1%, then just
              // print a zero to avoid printing silly things such as
              // "2.45e-6%". If not, then either print the actual percentage
              // or round up to 100.
              const double fraction = output_type == wall_times ?
                current_section.total_wall_time/total_time :
                current_section.total_cpu_time/total_time;
              if (fraction < 0.001)
                out_stream << 0.0;
              else if (fraction < 0.999)
                out_stream << std::fixed << std::setprecision(2) << fraction * 100;
              else
                out_stream << std::fixed << std::setprecision(1) << 100.0;


              out_stream << "% |";
            }
          else
            out_stream << 0.0 << "% |";
        }
      out_stream << '\n'
                 << "+---------------------------------+-----------+"
                 << "------------+------------+\n"
                 << '\n';

      if (output_type == cpu_times)
        {
          const double time_gap = cpu_check_time - timer_all.cpu_time();
          if (time_gap > 0.0)
            out_stream << std::endl
                       << "Note: The sum of counted times is " << time_gap
                       << " seconds larger than the total time.\n"
                       << "(Timer function may have introduced too much overhead, or different\n"
                       << "section timers may have run at the same time.)"
                       << std::endl;
        }
    };

  if (output_type == wall_times || output_type == cpu_and_wall_times)
    print_sections(wall_times);
  if (output_type == cpu_times || output_type == cpu_and_wall_times)
    print_sections(cpu_times);
}



void
TimerOutput::disable_output ()
{
  output_is_enabled = false;
}



void
TimerOutput::enable_output ()
{
  output_is_enabled = true;
}

void
TimerOutput::reset ()
{
  Threads::Mutex::ScopedLock lock (mutex);
  sections.clear();
  active_sections.clear();
  timer_all.restart();
}


DEAL_II_NAMESPACE_CLOSE
